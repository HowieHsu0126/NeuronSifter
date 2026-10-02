from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.migrations.legacy_checkpoint import migrate_legacy_checkpoint_keys
from Libs.models.modules.neuronsifter_world_action import NeuronSifterWorldActionOperator
from Libs.models.modules.quantized_mddm_rollout_operator import QuantizedMddmRolloutOperator
from Libs.models.modules.field_rollout import ActionConditionedOperatorBlock
from Libs.models.modules.transolver_operator import TransolverMicroenvironmentOperator
from Libs.utils.exp.experiment_tracking import seed_everything
from Libs.utils.model.dynamics_fields import node_adjacency_matrix
from Libs.utils.model.dynamics_store import load_chunked_rows, load_zarr_rows, read_zarr_array_metadata
from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, resolve_repo_path, write_json, write_text_atomic
from Libs.utils.data.occupancy_conditioning import load_conditioning_rows, read_conditioning_manifest
from Libs.utils.model.screening_model_config import quantized_mddm_rollout_from_config
from Libs.utils.model.neuronsifter_scale_registry import apply_neuronsifter_scale, default_checkpoint_scale, default_inference_profile, fsdp_profile_config
from Libs.utils.model.transolver_parameter_budget import neuronsifter_world_action_active_parameters, transolver_microenvironment_active_parameters, validate_parameter_budget


class MicroenvironmentZarrDataset(Dataset):
    def __init__(self, store_path: Path, config: dict, row_indices: list[int] | None = None, teacher_loss_weight: float = 1.0) -> None:
        self.store_path = store_path
        self.config = config
        self.teacher_loss_weight = float(teacher_loss_weight)
        backbone = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]
        self.conditioned = backbone["kind"] == "neuronsifter_world_action"
        self.conditioning_records = read_conditioning_manifest(store_path, config) if self.conditioned else {}
        if self.conditioned and row_indices is None:
            raise ValueError("Conditional training requires explicit QC-approved split indices, including smoke runs.")
        self.split_name = str(config["screening_pipeline"]["screening_benchmark"]["training"]["zarr_loader"]["training_split_name"])
        self.trajectory_metadata = read_zarr_array_metadata(store_path, "trajectories")
        self.uncertainty_metadata = read_zarr_array_metadata(store_path, "teacher_uncertainty")
        self.row_indices = list(range(int(self.trajectory_metadata["shape"][0]))) if row_indices is None else [int(index) for index in row_indices]
        if self.conditioned:
            for index in self.row_indices:
                if index not in self.conditioning_records or self.conditioning_records[index]["split_name"] != self.split_name:
                    raise ValueError("Conditioning selection is outside the requested split; no trajectory may be read.")
        self.sample_count = len(self.row_indices)

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.__getitems__([index])[0]

    def __getitems__(self, indices: list[int]) -> list[dict[str, torch.Tensor]]:
        row_indices = np.asarray([self.row_indices[int(index)] for index in indices], dtype=np.int64)
        trajectories = load_zarr_rows(self.store_path, "trajectories", row_indices)
        uncertainties = load_zarr_rows(self.store_path, "teacher_uncertainty", row_indices)
        conditions = load_conditioning_rows(
            self.store_path, row_indices, trajectories.shape, self.config,
            self.conditioning_records, self.split_name,
        ) if self.conditioned else {}
        return [
            {
                "trajectory": torch.tensor(trajectories[position], dtype=torch.float32),
                "uncertainty": torch.tensor(uncertainties[position], dtype=torch.float32),
                "teacher_loss_weight": torch.tensor(
                    self.teacher_loss_weight * (float(self.conditioning_records[int(row_indices[position])]["sample_weight"]) if self.conditioned else 1.0),
                    dtype=torch.float32,
                ),
                **{key: torch.as_tensor(value[position].copy()) for key, value in conditions.items()},
            }
            for position in range(len(indices))
        ]


class MicroenvironmentTeacherFamilyDataset(Dataset):
    def __init__(self, config: dict, split_name: str) -> None:
        self.config = config
        self.split_name = str(split_name)
        self.rows: list[dict[str, object]] = []
        self.parameter_prior_family_counts: dict[str, int] = {}
        self.conditioning_records: dict[Path, dict] = {}
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
            if self.config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]["kind"] == "neuronsifter_world_action":
                if row.get("qc_grade") not in {"A", "B"} or row.get("qc_use") != "main_pretraining" or not split_field:
                    raise ValueError("Every full-method teacher requires QC A/B and explicit split selection.")
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
                    "trajectory_uid": str(row.get("trajectory_uid", "")),
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.__getitems__([index])[0]

    def __getitems__(self, indices: list[int]) -> list[dict[str, torch.Tensor]]:
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
            conditioned = self.config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]["kind"] == "neuronsifter_world_action"
            if conditioned:
                if store_path not in self.conditioning_records:
                    self.conditioning_records[store_path] = read_conditioning_manifest(store_path, self.config)
                for output_position, row_index in positions:
                    record = self.conditioning_records[store_path].get(row_index)
                    expected_uid = self.rows[int(indices[output_position])]["trajectory_uid"]
                    if record is None or record["split_name"] != self.split_name or record["trajectory_uid"] != expected_uid:
                        raise ValueError("QC/conditioning identity or split mismatch before reading trajectory values.")
            if store_format == "neuronsifter_chunk_store":
                trajectories = load_chunked_rows(store_path, trajectory_array_name, row_indices)
                uncertainties = np.full((trajectories.shape[0], trajectories.shape[-1]), float(default_uncertainty_value), dtype=np.float32)
            elif store_format == "zarr_v2":
                trajectories = load_zarr_rows(store_path, trajectory_array_name, row_indices)
                uncertainties = load_zarr_rows(store_path, uncertainty_array_name, row_indices)
            else:
                raise ValueError(f"Unsupported teacher-family store format: {store_format}")
            conditioned = self.config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]["kind"] == "neuronsifter_world_action"
            conditions = {}
            if conditioned:
                if store_path not in self.conditioning_records:
                    self.conditioning_records[store_path] = read_conditioning_manifest(store_path, self.config)
                conditions = load_conditioning_rows(
                    store_path, row_indices, trajectories.shape, self.config,
                    self.conditioning_records[store_path], self.split_name,
                    [str(self.rows[int(indices[position])]["trajectory_uid"]) for position, _ in positions],
                )
            for local_position, (output_position, _) in enumerate(positions):
                row = self.rows[int(indices[output_position])]
                outputs[output_position] = {
                    "trajectory": torch.tensor(trajectories[local_position], dtype=torch.float32),
                    "uncertainty": torch.tensor(uncertainties[local_position], dtype=torch.float32),
                    "teacher_loss_weight": torch.tensor(float(row["teacher_loss_weight"]), dtype=torch.float32),
                    **{key: torch.as_tensor(value[local_position].copy()) for key, value in conditions.items()},
                }
        return [output for output in outputs if output is not None]


class MicroenvironmentPretrainingModel(nn.Module):
    def __init__(self, config: dict, use_smoke_dimensions: bool) -> None:
        super().__init__()
        backbone_config = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]
        species_count = len(config["dynamics_ready_registry"]["species"])
        hidden_dim = int(backbone_config["operator_hidden_dim"])
        depth = int(backbone_config["operator_depth"])
        head_count = 1
        if str(backbone_config["kind"]) in {"neuronsifter_world_action", "transolver_microenvironment_operator"}:
            head_count = int(backbone_config["attention_head_count"])
        if use_smoke_dimensions:
            smoke_config = config["screening_pipeline"]["screening_benchmark"]["training"]["zarr_loader"]
            hidden_dim = int(smoke_config.get("smoke_hidden_dim", 64))
            depth = int(smoke_config.get("smoke_depth", 2))
            head_count = int(smoke_config.get("smoke_head_count", 4))
        self.operator_kind = str(backbone_config["kind"])
        self.register_buffer("node_adjacency", torch.tensor(node_adjacency_matrix(config), dtype=torch.float32), persistent=False)
        self.state_encoder = nn.Linear(species_count, hidden_dim)
        self.condition_encoder = nn.Linear(species_count, hidden_dim)
        self.factor_operator_blocks = nn.ModuleList()
        if backbone_config["kind"] == "transolver_microenvironment_operator":
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
            mechanism_moe = backbone_config["mechanism_moe"]
            graph_config = dict(backbone_config["graph_spectral_tokenizer"])
            wavelet_config = dict(backbone_config["temporal_wavelet_tokenizer"])
            sampler_config = dict(backbone_config["diffusion_sampler"])
            self.operator = NeuronSifterWorldActionOperator(
                hidden_dim,
                depth,
                head_count,
                int(backbone_config["mechanism_state_token_count"]),
                int(backbone_config["node_coordinate_dim"]),
                float(backbone_config["operator_dropout_rate"]),
                [str(expert_name) for expert_name in mechanism_moe["expert_names"]],
                int(mechanism_moe["top_k"]),
                float(mechanism_moe["capacity_factor"]),
                float(mechanism_moe["entropy_loss_weight"]),
                float(mechanism_moe["load_balance_loss_weight"]),
                float(mechanism_moe["expert_capacity_loss_weight"]),
                int(mechanism_moe["expert_hidden_multiplier"]),
                int(backbone_config["dit_mlp_multiplier"]),
                int(graph_config["graph_mode_count"]),
                int(wavelet_config["wavelet_level_count"]),
                float(graph_config["eigenvalue_temperature"]),
                bool(graph_config["normalize_laplacian"]),
                float(graph_config["spectral_residual_scale"]),
                float(backbone_config["mechanism_slice_temperature"]),
                int(sampler_config["sample_count"]),
                int(sampler_config["denoising_step_count"]),
                int(sampler_config["sample_chunk_size"]),
                int(sampler_config["training_sample_count"]),
                int(sampler_config["training_denoising_step_count"]),
                float(sampler_config["diffusion_noise_scale"]),
                float(sampler_config["score_step_size"]),
                float(sampler_config["terminal_noise_scale"]),
                float(sampler_config["ranking_stability_scale"]),
                [float(value) for value in sampler_config["quantile_levels"]],
                int(sampler_config["sampler_seed"]),
                bool(sampler_config["return_sample_trajectories"]),
                dict(backbone_config["operator_splitting"]),
            )
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
        self.state_decoder = nn.Linear(hidden_dim, species_count)
        self.counterfactual_head = nn.Linear(hidden_dim, species_count)
        self.conditional_diffusion = None
        if self.operator_kind == "neuronsifter_world_action":
            sampler_config = quantized_mddm_rollout_from_config(config, backbone_config)
            if sampler_config is None or not sampler_config["enabled"]:
                raise ValueError("Full world-model pretraining requires the conditional wavelet diffusion sampler.")
            self.conditional_diffusion = QuantizedMddmRolloutOperator(
                sampler_config, species_count, hidden_dim, species_count,
                int(backbone_config["temporal_wavelet_tokenizer"]["wavelet_level_count"]),
            )
            # The full-horizon objective uses the sampler's representation and
            # decoder. Retired teacher-forced heads must not enter checkpoints.
            self.state_encoder = nn.Identity()
            self.condition_encoder = nn.Identity()
            self.state_decoder = nn.Identity()
            self.counterfactual_head = nn.Identity()

    def factor_coupled_diagnostics(self, encoded: torch.Tensor) -> dict[str, torch.Tensor]:
        zero = encoded.sum() * 0.0
        return {
            "expert_utilization": torch.ones(1, dtype=encoded.dtype, device=encoded.device),
            "router_entropy": zero,
            "router_regularization_loss": zero,
            "slice_entropy": zero,
        }

    def forward(self, state: torch.Tensor, coordinates: torch.Tensor,
                conditioned_inputs: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        if self.conditional_diffusion is not None:
            if conditioned_inputs is None:
                raise ValueError("Independent occupancy inputs are mandatory; future targets are never conditioning.")
            if state.ndim != 5:
                raise ValueError("Conditional training targets must be [batch,time,region,compartment,species].")
            output = self.conditional_diffusion(
                state[:, None], True,
                occupancy_program=conditioned_inputs["occupancy_program"][:, None],
                occupancy_support=conditioned_inputs["occupancy_support"][:, None],
                initial_state=state[:, None, 0],
                posterior_context=conditioned_inputs["posterior_context"][:, None],
                time_hours=conditioned_inputs["time_hours"][:, None],
                sample_keys=conditioned_inputs["sample_keys"][:, None],
                sample_weights=conditioned_inputs["sample_weights"][:, None],
                node_coordinates=coordinates,
                adjacency_matrix=self.node_adjacency.to(device=state.device),
                backbone=self.operator,
            )
            return {"diffusion_loss": output["rollout_auxiliary_loss"],
                    "state_prediction": output["trajectory_mean"][:, 0],
                    "router_diagnostics": output["sampler_diagnostics"]}
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
    parser.add_argument("--checkpoint-scale", default="")
    parser.add_argument("--inference-profile", default="")
    parser.add_argument("--distributed-strategy", default="")
    parser.add_argument("--one-batch-smoke", action="store_true")
    parser.add_argument("--single-batch-full-model-check", action="store_true")
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
    selected_seed = str(config.get("selected_formal_seed", config["screening_pipeline"]["screening_benchmark"]["training"]["random_seed"]))
    return artifact_root / subdir_template.format(
        benchmark_name=benchmark_name,
        profile_name=profile_name,
        seed=selected_seed,
    )


def manifest_row_indices(config: dict, store_path: Path, split_name: str) -> list[int]:
    families = config["screening_pipeline"]["screening_benchmark"]["training"]["teacher_family_training"]["families"]
    source_collection = str(config["screening_pipeline"]["screening_benchmark"]["training"]["zarr_loader"]["source_collection"])
    family = next((item for item in families if item["family_key"] == source_collection), None)
    if family is None or not family.get("qc_use_field") or not family.get("split_field"):
        raise ValueError("Direct-store pretraining requires a canonical QC-catalog family.")
    manifest_path = resolve_repo_path(family["catalog_path"])
    if not manifest_path.exists():
        return []
    store_relative = str(store_path.resolve().relative_to(REPO_ROOT))
    row_indices: list[int] = []
    conditioned = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]["kind"] == "neuronsifter_world_action"
    records = read_conditioning_manifest(store_path, config) if conditioned else {}
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["source_collection"] != source_collection:
                continue
            if row["split_name"] != split_name:
                continue
            if row["trajectory_store_path"] != store_relative:
                continue
            if row.get("qc_grade") not in {"A", "B"} or row.get("qc_use") != "main_pretraining":
                continue
            if conditioned:
                record = records.get(int(row["trajectory_index"]))
                if (record is None or record["trajectory_uid"] != row["trajectory_uid"] or record["split_name"] != split_name
                        or record["qc_grade"] != row["qc_grade"]
                        or float(record["sample_weight"]) != float(row["sample_weight"])):
                    raise ValueError("Independent conditioning does not match the canonical QC trajectory.")
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


def masked_reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor, mask_ratio: float, sample_weight: torch.Tensor) -> torch.Tensor:
    flat_index = torch.arange(target.shape[-2], device=target.device)
    period = max(int(round(1.0 / mask_ratio)), 1)
    mask = (flat_index % period == 0).view(1, 1, -1, 1)
    squared_error = (prediction - target) ** 2
    expanded_mask = mask.expand_as(squared_error)
    per_sample_loss = (squared_error * expanded_mask).reshape(prediction.shape[0], -1).sum(dim=1) / torch.clamp(
        expanded_mask.reshape(prediction.shape[0], -1).sum(dim=1),
        min=torch.finfo(prediction.dtype).eps,
    )
    return weighted_batch_mean(per_sample_loss, sample_weight)


def uncertainty_weighted_mse(prediction: torch.Tensor, target: torch.Tensor, uncertainty: torch.Tensor, config: dict, sample_weight: torch.Tensor) -> torch.Tensor:
    weighting = config["screening_pipeline"]["screening_benchmark"]["training"]["uncertainty_weighting"]
    precision = 1.0 / torch.clamp(uncertainty, min=float(weighting["minimum_teacher_precision"]))
    precision = torch.clamp(precision, min=float(weighting["minimum_teacher_precision"]), max=float(weighting["maximum_teacher_precision"]))
    while precision.ndim < prediction.ndim:
        precision = precision.unsqueeze(1)
    per_sample_loss = (precision * (prediction - target) ** 2).reshape(prediction.shape[0], -1).mean(dim=1)
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
) -> torch.Tensor:
    objective_config = config["neuronsifter_scale_registry"]["jepa_auxiliary_objectives"]["masked_species_group_prediction"]
    if not bool(objective_config["enabled"]):
        return zero_loss_like(prediction)
    selected_species = []
    for group_species in dict(objective_config["groups"]).values():
        selected_species.extend(str(species_name) for species_name in group_species)
    selected_indices = species_indices(config, selected_species)
    squared_error = (prediction[..., selected_indices] - target[..., selected_indices]) ** 2
    per_sample_loss = squared_error.reshape(prediction.shape[0], -1).mean(dim=1)
    return weighted_batch_mean(per_sample_loss, sample_weight)


def masked_compartment_prediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    config: dict,
    sample_weight: torch.Tensor,
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
    per_sample_loss = squared_error.reshape(prediction.shape[0], -1).mean(dim=1)
    return weighted_batch_mean(per_sample_loss, sample_weight)


def counterfactual_latent_delta_prediction_loss(
    counterfactual_prediction: torch.Tensor,
    state: torch.Tensor,
    target: torch.Tensor,
    config: dict,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    objective_config = config["neuronsifter_scale_registry"]["jepa_auxiliary_objectives"]["counterfactual_latent_delta_prediction"]
    if not bool(objective_config["enabled"]):
        return zero_loss_like(counterfactual_prediction)
    predicted_delta = counterfactual_prediction - state
    target_delta = target - state
    per_sample_loss = (predicted_delta - target_delta).reshape(state.shape[0], -1).pow(2).mean(dim=1)
    return weighted_batch_mean(per_sample_loss, sample_weight)


def rank_order_distillation_loss(prediction: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
    prediction_order = torch.softmax(prediction.reshape(prediction.shape[0], -1), dim=-1)
    target_order = torch.softmax(target.reshape(target.shape[0], -1), dim=-1)
    per_sample_loss = (prediction_order - target_order).pow(2).mean(dim=1)
    return weighted_batch_mean(per_sample_loss, sample_weight)


def uncertainty_calibration_distillation_loss(prediction: torch.Tensor, target: torch.Tensor, uncertainty: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
    predicted_dispersion = prediction.reshape(prediction.shape[0], -1).std(dim=1, unbiased=False)
    target_dispersion = target.reshape(target.shape[0], -1).std(dim=1, unbiased=False)
    teacher_uncertainty = uncertainty.reshape(uncertainty.shape[0], -1).mean(dim=1)
    per_sample_loss = (predicted_dispersion - (target_dispersion + teacher_uncertainty)).pow(2)
    return weighted_batch_mean(per_sample_loss, sample_weight)


def tri_consistency_distillation_loss(prediction: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor) -> torch.Tensor:
    predicted_reliability = torch.exp(-prediction.reshape(prediction.shape[0], -1).var(dim=1, unbiased=False))
    target_reliability = torch.exp(-target.reshape(target.shape[0], -1).var(dim=1, unbiased=False))
    return weighted_batch_mean((predicted_reliability - target_reliability).pow(2), sample_weight)


def expert_diagnostics_distillation_loss(router_diagnostics: dict[str, torch.Tensor]) -> torch.Tensor:
    utilization = router_diagnostics["expert_utilization"]
    target_utilization = torch.full_like(utilization, 1.0 / float(utilization.numel()))
    return (utilization - target_utilization).pow(2).mean()


def objective_loss(output: dict[str, torch.Tensor | dict[str, torch.Tensor]], batch: dict[str, torch.Tensor], config: dict) -> tuple[torch.Tensor, dict[str, float]]:
    if "diffusion_loss" in output:
        diagnostics = output["router_diagnostics"]
        loss = output["diffusion_loss"]
        router = diagnostics.get("router_regularization_loss")
        if router is None:
            raise ValueError("Conditional backbone must provide its router regularization diagnostic.")
        weight = config["screening_pipeline"]["screening_benchmark"]["training"]["pretraining_objectives"]["router_regularization_weight"]
        loss = loss + float(weight) * router
        return loss, {"conditional_diffusion_loss": float(output["diffusion_loss"].detach().cpu()),
                      "router_regularization_loss": float(router.detach().cpu()), "total_loss": float(loss.detach().cpu())}
    objectives = config["screening_pipeline"]["screening_benchmark"]["training"]["pretraining_objectives"]
    jepa_objectives = config["neuronsifter_scale_registry"]["jepa_auxiliary_objectives"]
    distillation_objectives = config["neuronsifter_scale_registry"]["distillation_losses"]
    trajectory = batch["trajectory"]
    uncertainty = batch["uncertainty"]
    state = trajectory[:, :-1].reshape(trajectory.shape[0], trajectory.shape[1] - 1, -1, trajectory.shape[-1])
    target_next = trajectory[:, 1:].reshape(trajectory.shape[0], trajectory.shape[1] - 1, -1, trajectory.shape[-1])
    prediction = output["state_prediction"]
    counterfactual_prediction = output["counterfactual_prediction"]
    sample_weight = batch["teacher_loss_weight"].to(device=prediction.device, dtype=prediction.dtype)
    reconstruction_loss = masked_reconstruction_loss(prediction, target_next, float(objectives["masked_state_ratio"]), sample_weight)
    next_state_loss = uncertainty_weighted_mse(prediction, target_next, uncertainty, config, sample_weight)
    long_horizon_loss = uncertainty_weighted_mse(prediction[:, -1:], target_next[:, -1:], uncertainty, config, sample_weight)
    distillation_loss = uncertainty_weighted_mse(prediction, target_next, uncertainty, config, sample_weight)
    counterfactual_loss = weighted_batch_mean((counterfactual_prediction - state).reshape(state.shape[0], -1).pow(2).mean(dim=1), sample_weight)
    router_diagnostics = output["router_diagnostics"]
    router_loss = router_diagnostics["router_regularization_loss"]
    species_group_loss = masked_species_group_prediction_loss(prediction, target_next, config, sample_weight)
    compartment_loss = masked_compartment_prediction_loss(prediction, target_next, config, sample_weight)
    latent_delta_loss = counterfactual_latent_delta_prediction_loss(counterfactual_prediction, state, target_next, config, sample_weight)
    rank_distillation_loss = rank_order_distillation_loss(prediction, target_next, sample_weight)
    uncertainty_distillation_loss = uncertainty_calibration_distillation_loss(prediction, target_next, uncertainty, sample_weight)
    tri_distillation_loss_value = tri_consistency_distillation_loss(prediction, target_next, sample_weight)
    expert_diagnostics_loss = expert_diagnostics_distillation_loss(router_diagnostics)
    distillation_terms = dict(distillation_objectives["terms"])
    total_loss = (
        float(objectives["masked_state_reconstruction_weight"]) * reconstruction_loss
        + float(objectives["next_state_prediction_weight"]) * next_state_loss
        + float(objectives["long_horizon_rollout_weight"]) * long_horizon_loss
        + float(objectives["teacher_distillation_weight"]) * distillation_loss
        + float(objectives["counterfactual_consistency_weight"]) * counterfactual_loss
        + float(objectives["router_regularization_weight"]) * router_loss
        + float(jepa_objectives["masked_species_group_prediction"]["weight"]) * species_group_loss
        + float(jepa_objectives["masked_compartment_prediction"]["weight"]) * compartment_loss
        + float(jepa_objectives["counterfactual_latent_delta_prediction"]["weight"]) * latent_delta_loss
        + float(distillation_terms["rank_order"]["weight"]) * rank_distillation_loss
        + float(distillation_terms["uncertainty_calibration"]["weight"]) * uncertainty_distillation_loss
        + float(distillation_terms["tri_consistency"]["weight"]) * tri_distillation_loss_value
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
        "total_loss": float(total_loss.detach().cpu()),
    }


def validate_default_parameter_budget(config: dict) -> int:
    backbone_config = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]
    budget = backbone_config["parameter_budget"]
    if backbone_config["kind"] == "transolver_microenvironment_operator":
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
    validate_parameter_budget(parameter_count, int(budget["active_parameter_lower_bound"]), int(budget["active_parameter_upper_bound"]))
    return parameter_count


def run_pretraining_step(
    model: MicroenvironmentPretrainingModel,
    batch: dict[str, torch.Tensor],
    config: dict,
    device: torch.device,
    memory_policy: dict[str, object],
) -> tuple[dict[str, float], dict[str, object]]:
    batch = {key: value.to(device) for key, value in batch.items()}
    trajectory = batch["trajectory"]
    conditioned = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]["kind"] == "neuronsifter_world_action"
    if conditioned:
        state = trajectory
        coordinates = node_coordinates(trajectory.shape[2], trajectory.shape[3], state.device).reshape(-1, 2)
        needed = {"occupancy_program", "occupancy_support", "posterior_context", "time_hours", "sample_keys"}
        if not needed.issubset(batch):
            raise ValueError("Full conditional pretraining batch lacks independent occupancy inputs.")
        uncertainty = batch["uncertainty"]
        if not torch.isfinite(uncertainty).all() or (uncertainty < 0).any():
            raise ValueError("Teacher uncertainty must be finite and nonnegative.")
        policy = config["screening_pipeline"]["screening_benchmark"]["training"]["uncertainty_weighting"]
        # Match teacher_precision_from_uncertainty's established registry policy.
        precision = uncertainty.mean(dim=-1).clamp_min(float(policy["minimum_teacher_precision"])).reciprocal().clamp(
            float(policy["minimum_teacher_precision"]), float(policy["maximum_teacher_precision"])
        )
        conditioned_inputs = {key: batch[key] for key in needed}
        conditioned_inputs["sample_weights"] = batch["teacher_loss_weight"] * precision
    else:
        state = trajectory[:, :-1].reshape(trajectory.shape[0], trajectory.shape[1] - 1, -1, trajectory.shape[-1])
        coordinates = node_coordinates(trajectory.shape[2], trajectory.shape[3], state.device).expand(state.shape[0], state.shape[1], -1, -1)
        conditioned_inputs = None
    with autocast_context(memory_policy, device):
        output = model(state, coordinates, conditioned_inputs)
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
    return loss_payload, router_payload


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
    return {
        parameter_name: parameter.detach().cpu()
        for parameter_name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def save_training_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    artifact_root: Path,
    checkpoint_name: str,
    epoch: int,
    executed_steps: int,
    distributed_strategy: str,
) -> Path:
    if distributed_strategy == "native_pytorch_fsdp":
        rank_suffix = f"rank_{distributed_rank()}"
        checkpoint_path = artifact_root / f"{checkpoint_name}_{rank_suffix}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": rank_local_model_shard(model),
                "optimizer_state_dict": cpu_checkpoint_payload(optimizer.state_dict()),
                "executed_steps": executed_steps,
                "checkpoint_policy": "rank_local_shard_state_dict",
                "distributed_rank": distributed_rank(),
            },
            checkpoint_path,
        )
        return checkpoint_path
    checkpoint_path = artifact_root / f"{checkpoint_name}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
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


def main() -> None:
    args = parse_args()
    raw_config = load_config(args.config)
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
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    parameter_count = validate_default_parameter_budget(config)
    registry = config["data"]["full_microenvironment_teacher_ready_registry"]
    relative_paths = registry["relative_paths"]
    store_path = resolve_repo_path(args.store) if args.store else resolve_repo_path(registry["local_root"]) / relative_paths["trajectory_store"]
    loader_config = config["screening_pipeline"]["screening_benchmark"]["training"]["zarr_loader"]
    single_batch_check = bool(args.one_batch_smoke or args.single_batch_full_model_check)
    memory_policy = training_memory_policy(config, single_batch_check)
    teacher_family_config = config["screening_pipeline"]["screening_benchmark"]["training"]["teacher_family_training"]
    if args.store or not bool(teacher_family_config["enabled"]):
        row_indices = manifest_row_indices(config, store_path, str(loader_config["training_split_name"]))
        dataset = MicroenvironmentZarrDataset(store_path, config, row_indices=row_indices)
    else:
        dataset = MicroenvironmentTeacherFamilyDataset(config, str(loader_config["training_split_name"]))
    if len(dataset) == 0:
        raise ValueError("Microenvironment pretraining dataset is empty.")
    batch_size = int(memory_policy["micro_batch_size"])
    device, distributed_strategy = initialize_device_and_distributed(args, fsdp_profile)
    training_seed = int(config["screening_pipeline"]["screening_benchmark"]["training"]["random_seed"])
    seed_everything(training_seed)
    sampler = DistributedSampler(dataset, shuffle=False) if distributed_strategy == "native_pytorch_fsdp" else None
    loader_kwargs = data_loader_kwargs(config)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler, **loader_kwargs)
    base_model = MicroenvironmentPretrainingModel(config, use_smoke_dimensions=False).to(device)
    trainable_parameter_count = sum(parameter.numel() for parameter in base_model.parameters() if parameter.requires_grad)
    if args.single_batch_full_model_check:
        budget = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]["parameter_budget"]
        validate_parameter_budget(trainable_parameter_count, int(budget["active_parameter_lower_bound"]), int(budget["active_parameter_upper_bound"]))
    model = wrap_fsdp_model(base_model, distributed_strategy, fsdp_profile, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["screening_pipeline"]["screening_benchmark"]["training"]["learning_rate"]), weight_decay=float(config["screening_pipeline"]["screening_benchmark"]["training"]["weight_decay"]))
    checkpoint_load_seconds = 0.0
    if args.resume_checkpoint:
        if distributed_strategy == "native_pytorch_fsdp":
            raise ValueError("Resume from checkpoint for native_pytorch_fsdp requires an explicit sharded load path and is not enabled for smoke validation.")
        checkpoint_load_started_at = time.perf_counter()
        checkpoint_payload = torch.load(resolve_repo_path(args.resume_checkpoint), map_location=device)
        checkpoint_load_seconds = time.perf_counter() - checkpoint_load_started_at
        state_dict = checkpoint_payload["model_state_dict"]
        if raw_operator_kind == "neuronsifter_world_action" and not any(key.startswith("conditional_diffusion.") for key in state_dict):
            raise ValueError("Unconditioned checkpoints cannot be resumed as the conditional wavelet method.")
        model.load_state_dict(state_dict if raw_operator_kind == "neuronsifter_world_action" else migrate_legacy_checkpoint_keys(state_dict))
        optimizer.load_state_dict(checkpoint_payload["optimizer_state_dict"])
    artifact_root = pretraining_artifact_root(config, single_batch_check)
    ensure_directory(artifact_root)
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
                "actual_trainable_parameter_count": trainable_parameter_count,
                "checkpoint_scale": checkpoint_scale,
                "inference_profile": inference_profile,
                "scale_selection": dict(config["neuronsifter_scale_selection"]),
                "model_dimension_policy": model_dimension_policy,
                "distributed_strategy": distributed_strategy,
                "fsdp_profile": fsdp_profile,
                "single_batch_full_model_check": bool(args.single_batch_full_model_check),
                "device": str(device),
                "training_seed": training_seed,
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
                "data_loader": loader_kwargs,
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
            "data_loader": loader_kwargs,
        }
        write_runtime_efficiency_profile(config, runtime_profile)
        complete_script_logging(
            pipeline_logger,
            pipeline_started_at,
            artifact=manifest_path,
            active_parameter_estimate=parameter_count,
            actual_trainable_parameter_count=trainable_parameter_count,
            checkpoint_scale=checkpoint_scale,
            inference_profile=inference_profile,
            device=str(device),
            total_loss=loss_payload["total_loss"],
        )
    finalize_distributed()


if __name__ == "__main__":
    main()
