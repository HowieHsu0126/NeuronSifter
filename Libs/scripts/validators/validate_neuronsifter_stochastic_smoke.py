from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import resource
import sys
import time

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.models.modules.neuronsifter_world_action import NeuronSifterWorldActionOperator
from Libs.models.modules.native_field_comparators import NativeWorldActionComparator
from Libs.exp.training.pretrain_microenvironment_world_model import MicroenvironmentPretrainingModel, NativeJointGroupDataset
from Libs.utils.data.native_diffusion_data import load_family_registries
from Libs.utils.data.native_input_staging import input_locations_from_config, stage_native_inputs
from Libs.utils.data.native_observable_profile import observable_profile
from Libs.utils.model.native_comparator_inputs import load_native_prediction_inputs
from Libs.utils.model.neuronsifter_scale_registry import apply_neuronsifter_scale
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, resolve_repo_path, write_json
from Libs.utils.model.dynamics_fields import node_adjacency_matrix
from Libs.utils.model.screening_model_config import rollout_update_policy_from_config


REQUIRED_OPERATOR_DIAGNOSTICS = [
    "operator_splitting_expert_utilization",
    "operator_splitting_router_entropy",
    "operator_splitting_species_attribution",
    "operator_splitting_coupling_matrix",
    "operator_splitting_score_norm_by_expert",
    "operator_splitting_uncertainty_by_expert",
    "operator_splitting_case_activation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the configured NeuronSifter stochastic UQ smoke.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--output", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--source-config", default="", help="Existing standalone public-source configuration; preparation only.")
    parser.add_argument("--stage-native-inputs", default="", help="Stage only selected public inputs and exit without model construction.")
    parser.add_argument("--native-input-manifest", default="")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested for NeuronSifter stochastic smoke, but CUDA is unavailable.")
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name != "cuda":
        raise ValueError(f"Unsupported NeuronSifter stochastic smoke device: {device_name}")
    return torch.device("cuda")


def node_coordinates(region_count: int, compartment_count: int, device: torch.device) -> torch.Tensor:
    region_denominator = max(region_count - 1, 1)
    compartment_denominator = max(compartment_count - 1, 1)
    coordinates = []
    for region_index in range(region_count):
        for compartment_index in range(compartment_count):
            coordinates.append(
                [
                    float(region_index) / float(region_denominator),
                    float(compartment_index) / float(compartment_denominator),
                ]
            )
    return torch.tensor(coordinates, dtype=torch.float32, device=device).view(1, 1, region_count * compartment_count, 2)


def finite_tensor(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all().item())


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def native_smoke_config(config: dict, source_config: dict | None = None) -> dict:
    """Select the canonical full architecture without changing source definitions."""
    if source_config is not None:
        if observable_profile(source_config) != observable_profile(config):
            raise ValueError("Public source and runtime quantity profiles/clocks differ")
        source = copy.deepcopy(source_config)
        source["screening_pipeline"]["screening_benchmark"] = copy.deepcopy(
            config["screening_pipeline"]["screening_benchmark"])
        source["neuronsifter_scale_registry"] = copy.deepcopy(config["neuronsifter_scale_registry"])
        config = source
    settings = config["screening_pipeline"]["screening_benchmark"]["training"]["gpu_stochastic_smoke"]
    policy = settings["native"]
    if (settings["use_smoke_dimensions"] or settings["mode"] != "native_full_runtime"
            or policy["checkpoint_scale"] != "neuronsifter_large"
            or (policy["complete_particles"], policy["conditional_draws_per_particle"], policy["denoising_steps"]) != (32, 2, 100)):
        raise ValueError("Native full runtime requires the original full architecture and32x2x100 budget")
    config = apply_neuronsifter_scale(config, policy["checkpoint_scale"], policy["inference_profile"])
    backbone = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]
    dimensions = tuple(backbone[key] for key in (
        "operator_hidden_dim", "operator_depth", "attention_head_count", "mechanism_state_token_count"))
    if (dimensions != (696, 8, 12, 96) or len(backbone["mechanism_moe"]["expert_names"]) != 10
            or backbone["graph_spectral_tokenizer"]["graph_mode_count"] != 36
            or backbone["temporal_wavelet_tokenizer"]["wavelet_level_count"] != 3):
        raise ValueError("Native full runtime architecture differs from696/8/12/96/10/36/3")
    profile = observable_profile(config)
    if profile is None or len(profile["quantities"]) != 26 or len(profile["families"]) != 4:
        raise ValueError("Native full runtime requires all26 quantities and four original domains")
    return config


def native_smoke_dataset(config: dict):
    policy = config["screening_pipeline"]["screening_benchmark"]["training"]["gpu_stochastic_smoke"]["native"]
    if policy["role"] != "development":
        raise ValueError("Runtime smoke reads only declared development public inputs")
    dataset = NativeJointGroupDataset(config, policy["role"],
        group_uids=set(policy["group_uids"]) if policy["group_uids"] else None)
    families = set(observable_profile(config)["families"])
    if dataset.family_group_counts != {family: 1 for family in families}:
        raise ValueError("Select exactly one complete public group per original domain")
    declarations = []
    for child, indices, _ in dataset.groups:
        rows = [child.native_records[index]["teacher"] for index in indices]
        row = rows[0]
        declarations.append({"sample_uid": row["sample_uid"], "posterior_uid": row["posterior_uid"],
            "family_uid": row["family_uid"], "source_uid": row["source_uid"], "role": policy["role"],
            "regimen_uids": [item["regimen_uid"] for item in rows],
            "software_fixture_only": all(item.get("software_fixture_only") is True for item in rows)})
    return dataset, declarations


class NativeRuntimeWitness:
    """Observe the existing sampler, including its actual learned module calls."""
    def __init__(self, model, report: dict, progress=None):
        self.model, self.report = model, report
        self.progress = progress
        self.calls = {"wrapper": 0, "sampler": 0, "backbone": 0,
                      "blocks": [0] * len(model.operator.blocks), "terminal_paths": 0}
        self.report["actual_calls"] = self.calls
        self.handles = []

    def __enter__(self):
        def backbone(module, args):
            self.calls["backbone"] += 1
            histogram = self.report.setdefault("neural_batch_size_histogram", {})
            size = str(args[0].shape[0])
            histogram[size] = histogram.get(size, 0) + 1
        def block(index):
            def record(module, args):
                self.calls["blocks"][index] += 1
            return record
        def before(module, args, kwargs):
            if args[1] or kwargs.get("teacher_trajectory") is not None or kwargs.get("teacher_confidence") is not None:
                raise ValueError("Runtime smoke must never construct or consume teacher tensors")
            self.calls["sampler"] += 1
        def after(module, args, kwargs, output):
            samples = output["sample_trajectories"]
            original = kwargs["initial_state"]
            if not torch.equal(samples[:, :, :, 0], original[None].expand_as(samples[:, :, :, 0])):
                raise ValueError("Native sampler changed the original public initial snapshot")
            for name in ("sample_trajectories", "trajectory_mean", "trajectory_std", "trajectory_quantiles", "ranking_stability"):
                if not finite_tensor(output[name]):
                    raise ValueError("Nonfinite native sampling output: " + name)
            for name in REQUIRED_OPERATOR_DIAGNOSTICS:
                if name not in output["sampler_diagnostics"] or not finite_tensor(output["sampler_diagnostics"][name]):
                    raise ValueError("Missing/nonfinite executed native diagnostic: " + name)
            if not torch.equal(output["quantity_support"], kwargs["conditioning"]["quantity_support"]):
                raise ValueError("Native sampler changed quantity support")
            self.report["finite_t0_support_checks"] = "passed"
            if self.calls["sampler"] == 1 and self.progress:
                self.report["progress_event"] = "first_actual_sampler_completed"
                self.progress(self.report)
        self.reverse = self.model.quantized_mddm.reverse_step
        def reverse(tokens, logits, u, v, generator=None, support=None):
            result = self.reverse(tokens, logits, u, v, generator, support)
            # The implementation recursively processes supported cells. Count
            # only the outer complete-grid invocation, once per sampled path.
            if v == 0 and support is not None:
                self.calls["terminal_paths"] += 1
                if (result == self.model.quantized_mddm.codebook_size).any():
                    raise ValueError("Native terminal categorical mask was not exhausted")
            return result
        self.model.quantized_mddm.reverse_step = reverse
        self.handles.append(self.model.operator.register_forward_pre_hook(backbone))
        self.handles.extend(module.register_forward_pre_hook(block(index)) for index, module in enumerate(self.model.operator.blocks))
        self.handles.append(self.model.quantized_mddm.register_forward_pre_hook(before, with_kwargs=True))
        self.handles.append(self.model.quantized_mddm.register_forward_hook(after, with_kwargs=True))
        return self

    def __exit__(self, *exception):
        self.model.quantized_mddm.reverse_step = self.reverse
        for handle in self.handles:
            handle.remove()


class NativeSmokeBelief:
    def __init__(self, model):
        self.model = model
        self.field_normalization = model.field_normalization

    def __call__(self, source):
        return self.model.infer_joint_source(source)


def native_runtime_groups(model, config: dict, declarations: list[dict], report: dict, progress=None) -> None:
    """Exercise the existing all-particle predictor; no target dataset indexing."""
    policy = config["screening_pipeline"]["screening_benchmark"]["training"]["gpu_stochastic_smoke"]["native"]
    registries = load_family_registries(config)
    predictor = NativeWorldActionComparator(model, config)
    belief = NativeSmokeBelief(model)
    draws, steps = policy["conditional_draws_per_particle"], policy["denoising_steps"]
    seed = model.quantized_mddm.sampler_seed
    expected = {"wrapper": 0, "sampler": 0, "backbone": 0, "terminal_paths": 0}
    report["groups"] = []
    with torch.no_grad(), NativeRuntimeWitness(model, report, progress) as witness:
        for declaration in declarations:
            report["current_group"] = declaration
            inputs = load_native_prediction_inputs(config, declaration, belief, registries)
            b, particles, actions = inputs.group_shape
            if b != 1 or particles != policy["complete_particles"] or set(inputs.regimen_uids) != set(declaration["regimen_uids"]):
                raise ValueError("Native smoke requires the complete declared tuple/action group")
            model.field_normalization.assert_support(inputs.posterior.support)
            model.field_normalization.assert_representation(model.operator, model.quantized_mddm)
            witness.calls["wrapper"] += 1
            output = predictor.sample_native(inputs, model.field_normalization, draws, steps, seed)
            if (output["particle_trajectory_samples"].shape[:4] != (draws, b, particles, actions)
                    or output["forecast_paths_per_action"] != particles * draws
                    or not torch.equal(output["particle_weights"], inputs.weights)):
                raise ValueError("Native wrapper changed complete-particle sampling axes or weights")
            initial = output["particle_trajectory_samples"][:, :, :, :, 0]
            public_initial = inputs.posterior.support.initial_field[None, :, :, None].expand_as(initial)
            support = inputs.validity.reshape(b, particles, actions, *inputs.validity.shape[1:])[:, :, :, 0]
            supported = support[None].expand_as(initial)
            difference = (initial[supported] - public_initial[supported]).abs().max().item()
            if not torch.equal(initial[supported], public_initial[supported]):
                raise ValueError("Native wrapper changed the public physical t0 under original support; max difference=" + str(difference))
            expected["wrapper"] += 1
            expected["sampler"] += b * particles
            expected["terminal_paths"] += b * particles * actions * draws
            sampler = model.quantized_mddm
            batch_width = (sampler.sample_chunk_size if seed >= 0 and not sampler.training
                           and not model.operator.training else 1)
            expected["backbone"] += particles * ((b * actions * draws + batch_width - 1) // batch_width) * steps
            report["groups"].append({**declaration, "particle_uids": inputs.particle_uids,
                "sample_uids": inputs.sample_uids, "regimen_uids": inputs.regimen_uids,
                "particle_count": particles, "action_count": actions, "draws_per_particle": draws,
                "forecast_paths_per_action": particles * draws, "denoising_steps": steps,
                "time_hours": inputs.grid["time_hours"], "quantity_count": inputs.validity.shape[-1],
                "normalization_id": inputs.normalization_id, "sampler_seed": seed,
                "physical_t0_max_abs_difference": difference,
                "particle_stream_stride": config["native_scientific_training"]["particle_stream_stride"],
                "rng_sample_ids": list(range(b)), "rng_regimen_ids": list(range(actions)), "draw_offset": 0,
                "sample_shape": list(output["particle_trajectory_samples"].shape)})
            if progress:
                report["progress_event"] = "complete_domain_sampled"
                progress(report)
            del inputs, output
    report["expected_calls"] = {**expected, "blocks": [expected["backbone"]] * len(model.operator.blocks)}
    if report["actual_calls"] != report["expected_calls"]:
        raise ValueError("Executed native module or terminal-path counts differ from complete workload")


def native_stochastic_smoke_payload(config: dict, device: torch.device, report: dict, progress=None) -> dict:
    locations = input_locations_from_config(config)
    if locations is None or locations.stores:
        raise ValueError("Native runtime requires an explicitly staged public-only input manifest")
    _, declarations = native_smoke_dataset(config)
    report.update(schema_version="native_stochastic_smoke.v1", smoke_status="started",
        weights_status="randomly_initialized", scientific_quality="NOT_EVALUATED", publication_eligible=False,
        qualification_scope="full_native_runtime_only", future_target_array_reads=0,
        teacher_loss_enabled=False, sampling_shape_carrier="zero_tensor_without_targets",
        source_scope="retained_software_fixture" if all(row["software_fixture_only"] for row in declarations) else "declared_public_source",
        device=str(device), initialization_seed=int(config["screening_pipeline"]["screening_benchmark"]["training"]["random_seed"]),
        input_manifest=str(locations.root / "native_inputs.json"),
        slurm_job_id=os.environ.get("SLURM_JOB_ID", ""),
        slurm_job_name=os.environ.get("SLURM_JOB_NAME", ""))
    torch.set_num_threads(1)
    torch.manual_seed(int(config["screening_pipeline"]["screening_benchmark"]["training"]["random_seed"]))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    def snapshot(value):
        value["resources"] = {"elapsed_seconds": time.perf_counter() - started,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
            "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
            "torch_threads": torch.get_num_threads()}
        if progress:
            progress(value)
    try:
        report["execution_phase"] = "model_initialization"
        snapshot(report)
        model = MicroenvironmentPretrainingModel(config, use_smoke_dimensions=False).to(device).eval()
        native, sampler = model.operator, model.quantized_mddm
        if (native.hidden_dim != 696 or native.depth != 8 or len(native.blocks) != 8 or native.mechanism_token_mixer.token_count != 96
                or any(block.self_attention.num_heads != 12 or block.cross_attention.num_heads != 12
                       or len(block.operator_splitting_expert_bank.experts) != 10 for block in native.blocks)
                or native.tokenizer.graph_mode_count != 36 or native.tokenizer.wavelet_level_count != 3
                or sampler.hidden_dim != 696 or sampler.species_count != 26 or not model.field_normalization.active):
            raise ValueError("Constructed full native architecture/normalizer differs from required runtime")
        report.update(model_parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            native_backbone_parameters=sum(p.numel() for p in native.parameters()),
            normalization_id=model.field_normalization.identity,
            architecture={"hidden": native.hidden_dim, "blocks": len(native.blocks), "heads": 12,
                          "mechanism_tokens": 96, "experts": 10, "graph_modes": 36, "haar_levels": 3, "quantities": 26})
        report["execution_phase"] = "native_sampling"
        snapshot(report)
        native_runtime_groups(model, config, declarations, report, snapshot)
        report["smoke_status"] = "passed"
        return report
    finally:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        snapshot(report)


def stochastic_smoke_payload(config: dict, device: torch.device) -> dict[str, object]:
    benchmark = config["screening_pipeline"]["screening_benchmark"]
    training_config = benchmark["training"]
    smoke_config = training_config["gpu_stochastic_smoke"]
    transition_backbone = dict(rollout_update_policy_from_config(config)["transition_backbone"])
    sampler_config = dict(transition_backbone["diffusion_sampler"])
    graph_config = dict(transition_backbone["graph_spectral_tokenizer"])
    wavelet_config = dict(transition_backbone["temporal_wavelet_tokenizer"])
    mechanism_moe = dict(transition_backbone["mechanism_moe"])
    operator_splitting = dict(transition_backbone["operator_splitting"])
    hidden_dim = int(transition_backbone["operator_hidden_dim"])
    depth = int(transition_backbone["operator_depth"])
    head_count = int(transition_backbone["attention_head_count"])
    if bool(smoke_config["use_smoke_dimensions"]):
        zarr_loader = training_config["zarr_loader"]
        hidden_dim = int(zarr_loader["smoke_hidden_dim"])
        depth = int(zarr_loader["smoke_depth"])
        head_count = int(zarr_loader["smoke_head_count"])
    torch.manual_seed(int(training_config["random_seed"]))
    region_count = len(config["spatial_atlas_registry"]["regions"])
    compartment_count = len(config["screening_pipeline"]["pkpd_ready_registry"]["compartments"])
    species_count = len(config["dynamics_ready_registry"]["species"])
    batch_size = int(smoke_config["batch_size"])
    regimen_count = int(smoke_config["regimen_count"])
    time_step_count = int(smoke_config["time_step_count"])
    adjacency = torch.tensor(node_adjacency_matrix(config), dtype=torch.float32, device=device)
    coordinates = node_coordinates(region_count, compartment_count, device).expand(batch_size, time_step_count, -1, -1)
    node_state = torch.randn(
        batch_size,
        time_step_count,
        region_count * compartment_count,
        hidden_dim,
        dtype=torch.float32,
        device=device,
    )
    conditioning_state = torch.randn_like(node_state)
    operator = NeuronSifterWorldActionOperator(
        hidden_dim,
        depth,
        head_count,
        int(transition_backbone["mechanism_state_token_count"]),
        int(transition_backbone["node_coordinate_dim"]),
        0.0,
        [str(expert_name) for expert_name in mechanism_moe["expert_names"]],
        int(mechanism_moe["top_k"]),
        float(mechanism_moe["capacity_factor"]),
        float(mechanism_moe["entropy_loss_weight"]),
        float(mechanism_moe["load_balance_loss_weight"]),
        float(mechanism_moe["expert_capacity_loss_weight"]),
        int(mechanism_moe["expert_hidden_multiplier"]),
        int(transition_backbone["dit_mlp_multiplier"]),
        int(graph_config["graph_mode_count"]),
        int(wavelet_config["wavelet_level_count"]),
        float(graph_config["eigenvalue_temperature"]),
        bool(graph_config["normalize_laplacian"]),
        float(graph_config["spectral_residual_scale"]),
        float(transition_backbone["mechanism_slice_temperature"]),
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
        operator_splitting,
    ).to(device)
    operator.eval()
    prior_trajectory = torch.sigmoid(
        torch.randn(
            batch_size,
            regimen_count,
            time_step_count,
            region_count,
            compartment_count,
            species_count,
            dtype=torch.float32,
            device=device,
        )
    )
    with torch.no_grad():
        _, operator_diagnostics = operator(node_state, conditioning_state, coordinates, adjacency)
        diffusion_payload = operator.trajectory_diffusion_summary(prior_trajectory, False, adjacency)
    sampler_diagnostics = diffusion_payload["sampler_diagnostics"]
    if not isinstance(sampler_diagnostics, dict):
        raise TypeError("NeuronSifter stochastic smoke expected sampler diagnostics.")
    missing_operator_diagnostics = [
        diagnostic_name
        for diagnostic_name in REQUIRED_OPERATOR_DIAGNOSTICS
        if diagnostic_name not in operator_diagnostics
    ]
    finite_checks = {
        "trajectory_std": finite_tensor(diffusion_payload["trajectory_std"]),
        "trajectory_quantiles": finite_tensor(diffusion_payload["trajectory_quantiles"]),
        "ranking_stability": finite_tensor(diffusion_payload["ranking_stability"]),
        "operator_splitting_diagnostics": all(
            finite_tensor(operator_diagnostics[name])
            for name in REQUIRED_OPERATOR_DIAGNOSTICS
            if name in operator_diagnostics
        ),
        "sampler_diagnostics": all(
            finite_tensor(value)
            for value in sampler_diagnostics.values()
            if isinstance(value, torch.Tensor)
        ),
    }
    observed_sample_count = int(round(scalar(sampler_diagnostics["stochastic_sample_count"])))
    observed_denoising_step_count = int(round(scalar(sampler_diagnostics["denoising_step_count"])))
    passed = (
        observed_sample_count == int(sampler_config["sample_count"])
        and observed_denoising_step_count == int(sampler_config["denoising_step_count"])
        and not missing_operator_diagnostics
        and all(finite_checks.values())
    )
    return {
        "smoke_status": "passed" if passed else "failed",
        "qualification_scope": "legacy_random_residual_diagnostic_not_native_learned_sampling",
        "weights_status": "randomly_initialized",
        "scientific_quality": "NOT_EVALUATED",
        "publication_eligible": False,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME", ""),
        "operator_kind": "neuronsifter_world_action",
        "operator_scale": "smoke_dimensions_formal_sampler_budget" if bool(smoke_config["use_smoke_dimensions"]) else "configured_operator_dimensions",
        "batch_size": batch_size,
        "regimen_count": regimen_count,
        "time_step_count": time_step_count,
        "region_count": region_count,
        "compartment_count": compartment_count,
        "species_count": species_count,
        "required_sample_count": int(sampler_config["sample_count"]),
        "observed_sample_count": observed_sample_count,
        "required_denoising_step_count": int(sampler_config["denoising_step_count"]),
        "observed_denoising_step_count": observed_denoising_step_count,
        "quantile_levels": [float(value) for value in sampler_config["quantile_levels"]],
        "finite_checks": finite_checks,
        "trajectory_std_mean": scalar(diffusion_payload["trajectory_std"].mean()),
        "trajectory_quantile_mean": scalar(diffusion_payload["trajectory_quantiles"].mean()),
        "ranking_stability_mean": scalar(diffusion_payload["ranking_stability"].mean()),
        "sampler_diagnostics": {
            key: scalar(value)
            for key, value in sampler_diagnostics.items()
            if isinstance(value, torch.Tensor) and value.numel() == 1
        },
        "required_operator_diagnostics": REQUIRED_OPERATOR_DIAGNOSTICS,
        "missing_operator_diagnostics": missing_operator_diagnostics,
        "operator_splitting_diagnostic_shapes": {
            name: list(operator_diagnostics[name].shape)
            for name in REQUIRED_OPERATOR_DIAGNOSTICS
            if name in operator_diagnostics
        },
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    smoke_config = config["screening_pipeline"]["screening_benchmark"]["training"]["gpu_stochastic_smoke"]
    if not bool(smoke_config["enabled"]):
        raise ValueError("screening_benchmark.training.gpu_stochastic_smoke.enabled must be true for formal smoke.")
    mode = smoke_config.get("mode", "legacy_residual_diagnostic")
    if mode == "native_full_runtime":
        if args.source_config and not args.stage_native_inputs:
            raise ValueError("--source-config is preparation only; run with the materialized staged config")
        config = native_smoke_config(config, load_config(args.source_config) if args.source_config else None)
        smoke_config = config["screening_pipeline"]["screening_benchmark"]["training"]["gpu_stochastic_smoke"]
        if args.stage_native_inputs:
            if args.native_input_manifest:
                raise ValueError("Public staging requires original inputs, not another manifest")
            dataset, declarations = native_smoke_dataset(config)
            manifest = stage_native_inputs(config, [dataset], args.stage_native_inputs, include_targets=False)
            # Standalone config preserves all original source definitions and
            # logical provenance paths. The manifest supplies only local copies.
            config_path = manifest.parent / "native_smoke_config.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            print(json.dumps({"native_input_manifest": str(manifest), "config": str(config_path),
                "input_scope": "public_prediction_only", "groups": declarations,
                "model_initializations": 0, "source_generation": 0, "normalization_fits": 0}))
            return
        if args.native_input_manifest:
            config["native_input_staging"] = {"manifest_path": args.native_input_manifest}
    elif mode != "legacy_residual_diagnostic" or args.source_config or args.stage_native_inputs or args.native_input_manifest:
        raise ValueError("Unsupported smoke mode or native options supplied to the legacy diagnostic")
    device = resolve_device(args.device or str(smoke_config["device"]))
    output_path = resolve_repo_path(args.output or str(smoke_config["output_path"]))
    ensure_directory(output_path.parent)
    payload = {"smoke_status": "started", "publication_eligible": False,
               "scientific_quality": "NOT_EVALUATED", "weights_status": "randomly_initialized"}
    try:
        if mode == "native_full_runtime":
            payload = native_stochastic_smoke_payload(config, device, payload,
                progress=lambda value: write_json(output_path, value))
        else:
            payload = stochastic_smoke_payload(config, device)
    except BaseException as error:
        payload.update(smoke_status="failed", error=type(error).__name__ + ": " + str(error),
            failure_phase=payload.get("execution_phase", "configuration_or_input"),
            failure_kind="resource" if isinstance(error, (MemoryError, torch.cuda.OutOfMemoryError)) else "runtime_or_contract")
        raise
    finally:
        write_json(output_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=False))
    if payload["smoke_status"] != str(smoke_config["required_status"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
