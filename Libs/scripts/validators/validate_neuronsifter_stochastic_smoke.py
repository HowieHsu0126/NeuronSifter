from __future__ import annotations

# Standalone script bootstraps the repository path before project imports.
# ruff: noqa: E402

import argparse
import json
import os
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.exp.training.pretrain_microenvironment_world_model import MicroenvironmentPretrainingModel
from Libs.scripts.reports.audit_world_action_manuscript import analytic_conditioned_batch
from Libs.utils.data.raw_data_registry import (
    ensure_directory,
    load_config,
    resolve_repo_path,
    write_json,
)


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
    parser = argparse.ArgumentParser(
        description="Run the configured NeuronSifter stochastic UQ smoke."
    )
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--output", default="")
    parser.add_argument("--device", default="")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA requested for NeuronSifter stochastic smoke, but CUDA is unavailable."
        )
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name != "cuda":
        raise ValueError(
            f"Unsupported NeuronSifter stochastic smoke device: {device_name}"
        )
    return torch.device("cuda")


def node_coordinates(
    region_count: int, compartment_count: int, device: torch.device
) -> torch.Tensor:
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
    return torch.tensor(coordinates, dtype=torch.float32, device=device).view(
        1, 1, region_count * compartment_count, 2
    )


def finite_tensor(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all().item())


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def stochastic_smoke_payload(config: dict, device: torch.device) -> dict[str, object]:

    torch.set_num_threads(1)
    benchmark = config["screening_pipeline"]["screening_benchmark"]
    smoke = benchmark["training"]["gpu_stochastic_smoke"]
    budget = benchmark["model_architecture"]["rollout_operator"]["transition_backbone"][
        "diffusion_sampler"
    ]
    torch.manual_seed(int(benchmark["training"]["random_seed"]))
    model = (
        MicroenvironmentPretrainingModel(config, bool(smoke["use_smoke_dimensions"]))
        .to(device)
        .eval()
    )
    b, a, t = (
        int(smoke["batch_size"]),
        int(smoke["regimen_count"]),
        int(smoke["time_step_count"]),
    )
    batch = {
        key: value.to(device)
        for key, value in analytic_conditioned_batch(config, b * a, t).items()
    }
    r, c, s = batch["trajectory"].shape[2:]
    output = model.conditional_diffusion.sample(
        initial_state=batch["trajectory"][:, 0].reshape(b, a, r, c, s),
        occupancy_program=batch["occupancy_program"].reshape(b, a, t, r, c, -1),
        occupancy_support=batch["occupancy_support"].reshape(b, a, t, r, c, -1),
        posterior_context=batch["posterior_context"].reshape(b, a, -1),
        time_hours=batch["time_hours"].reshape(b, a, t),
        sample_keys=batch["sample_keys"].reshape(b, a),
        node_coordinates=node_coordinates(r, c, device).reshape(-1, 2),
        adjacency_matrix=model.node_adjacency,
        backbone=model.operator,
        sample_count=int(budget["sample_count"]),
        denoising_steps=int(budget["denoising_step_count"]),
    )
    diagnostics = output["sampler_diagnostics"]
    missing = [key for key in REQUIRED_OPERATOR_DIAGNOSTICS if key not in diagnostics]
    finite_checks = {
        key: finite_tensor(output[key])
        for key in (
            "trajectory_mean",
            "trajectory_std",
            "trajectory_quantiles",
            "ranking_stability",
        )
    }
    sample_count = int(scalar(diagnostics["mddm_sample_count"]))
    steps = int(scalar(diagnostics["mddm_denoising_step_count"]))
    passed = (
        all(finite_checks.values())
        and not missing
        and scalar(diagnostics["mddm_final_unresolved_mask_fraction"]) == 0
    )
    return {
        "smoke_status": "passed" if passed else "failed",
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME", ""),
        "operator_kind": "neuronsifter_world_action",
        "main_sampler_kind": "occupancy_conditioned_masked_wavelet_diffusion",
        "operator_scale": "smoke_dimensions_formal_sampler_budget"
        if smoke["use_smoke_dimensions"]
        else "configured_operator_dimensions",
        "batch_size": b,
        "regimen_count": a,
        "time_step_count": t,
        "region_count": r,
        "compartment_count": c,
        "species_count": s,
        "required_sample_count": int(budget["sample_count"]),
        "observed_sample_count": sample_count,
        "required_denoising_step_count": int(budget["denoising_step_count"]),
        "observed_denoising_step_count": steps,
        "quantile_levels": model.conditional_diffusion.quantile_levels,
        "finite_checks": finite_checks,
        "trajectory_std_mean": scalar(output["trajectory_std"].mean()),
        "trajectory_quantile_mean": scalar(output["trajectory_quantiles"].mean()),
        "ranking_stability_mean": scalar(output["ranking_stability"].mean()),
        "sampler_diagnostics": {
            key: scalar(value)
            for key, value in diagnostics.items()
            if isinstance(value, torch.Tensor) and value.numel() == 1
        },
        "required_operator_diagnostics": REQUIRED_OPERATOR_DIAGNOSTICS,
        "missing_operator_diagnostics": missing,
        "operator_splitting_diagnostic_shapes": {
            key: list(diagnostics[key].shape)
            for key in REQUIRED_OPERATOR_DIAGNOSTICS
            if key in diagnostics
        },
        "uncertainty_status": output["uncertainty_status"],
        "publication_eligible": False,
        "qualification_scope": "engineering_only_untrained_full_architecture_no_biological_data",
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    smoke_config = config["screening_pipeline"]["screening_benchmark"]["training"][
        "gpu_stochastic_smoke"
    ]
    if not bool(smoke_config["enabled"]):
        raise ValueError(
            "screening_benchmark.training.gpu_stochastic_smoke.enabled must be true for formal smoke."
        )
    device = resolve_device(args.device or str(smoke_config["device"]))
    output_path = resolve_repo_path(args.output or str(smoke_config["output_path"]))
    payload = stochastic_smoke_payload(config, device)
    ensure_directory(output_path.parent)
    write_json(output_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=False))
    if payload["smoke_status"] != str(smoke_config["required_status"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
