"""Bounded conditional world-model training/sampling qualification.

Uses explicit analytic action inputs and synthetic responses. These checks never
qualify biological performance, a publication checkpoint, or empirical claims.
"""

from __future__ import annotations

# Standalone script bootstraps the repository path before project imports.
# ruff: noqa: E402

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.exp.training.pretrain_microenvironment_world_model import (
    MicroenvironmentPretrainingModel,
    node_coordinates,
    run_pretraining_step,
)
from Libs.utils.data.raw_data_registry import load_config, write_json


def analytic_conditioned_batch(
    config: dict, count: int = 2, time_count: int = 5
) -> dict:
    r = len(config["spatial_atlas_registry"]["regions"])
    c = len(config["screening_pipeline"]["pkpd_ready_registry"]["compartments"])
    s = len(config["dynamics_ready_registry"]["species"])
    hours = torch.linspace(0, 8, time_count).expand(count, -1).clone()
    # Occupancy is computed before the response from dose/affinity and clearance.
    dose = torch.linspace(0.2, 0.8, count)[:, None]
    concentration = dose * torch.exp(-0.15 * hours)
    occupancy = concentration / (concentration + 0.5)
    occupancy = occupancy[:, :, None, None, None].expand(-1, -1, r, c, 1).clone()
    initial = torch.full((count, r, c, s), 0.6)
    elapsed = hours[:, :, None, None, None] / 8
    response = elapsed * occupancy * torch.linspace(-0.25, 0.3, s)
    return dict(
        trajectory=initial[:, None] + response,
        occupancy_program=occupancy,
        occupancy_support=torch.ones_like(occupancy, dtype=torch.bool),
        posterior_context=torch.full((count, s), 0.1),
        time_hours=hours,
        sample_keys=torch.arange(5001, 5001 + count, dtype=torch.int64),
        teacher_loss_weight=torch.ones(count),
        uncertainty=torch.full((count, s), 0.1),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--optimization-steps", type=int, default=24)
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--denoising-steps", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    device = torch.device(args.device)
    config = load_config(args.config)
    torch.manual_seed(3407)
    model = MicroenvironmentPretrainingModel(config, True).to(device)
    batch = {
        key: value.to(device)
        for key, value in analytic_conditioned_batch(config).items()
    }
    r, c = batch["trajectory"].shape[2:4]
    coords = node_coordinates(r, c, device).reshape(-1, 2)
    inputs = {
        key: value
        for key, value in batch.items()
        if key not in {"trajectory", "uncertainty", "teacher_loss_weight"}
    }
    inputs["sample_weights"] = batch["teacher_loss_weight"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    memory_policy = {
        "gradient_accumulation_steps": 1,
        "mixed_precision": {
            "enabled": False,
            "device_type": args.device,
            "dtype": "float32",
        },
    }

    def fixed_corruption_loss():
        model.eval()
        torch.manual_seed(9021)
        with torch.no_grad():
            return float(
                model(batch["trajectory"], coords, inputs)["diffusion_loss"]
                .detach()
                .cpu()
            )

    started = time.perf_counter()
    before = fixed_corruption_loss()
    losses = []
    for index in range(args.optimization_steps):
        model.train()
        torch.manual_seed(12000 + index)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = run_pretraining_step(model, batch, config, device, memory_policy)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss["total_loss"])
    after = fixed_corruption_loss()
    model.eval()
    conditions = dict(
        initial_state=batch["trajectory"][:1, None, 0],
        occupancy_program=batch["occupancy_program"][:1, None],
        occupancy_support=batch["occupancy_support"][:1, None],
        posterior_context=batch["posterior_context"][:1, None],
        time_hours=batch["time_hours"][:1, None],
        sample_keys=batch["sample_keys"][:1, None],
        node_coordinates=coords,
        adjacency_matrix=model.node_adjacency,
        backbone=model.operator,
        sample_count=args.sample_count,
        denoising_steps=args.denoising_steps,
    )
    prediction = model.conditional_diffusion.sample(**conditions)
    if device.type == "cuda":
        torch.cuda.synchronize()
    checks = {
        "finite_optimization": all(torch.isfinite(torch.tensor(losses))),
        "fixed_corruption_loss_reduced": after < before,
        "all_masks_resolved": float(
            prediction["sampler_diagnostics"][
                "mddm_final_unresolved_mask_fraction"
            ].cpu()
        )
        == 0,
        "finite_samples": bool(torch.isfinite(prediction["trajectory_mean"]).all()),
        "initial_state_preserved": bool(
            torch.allclose(
                prediction["trajectory_mean"][:, :, 0], conditions["initial_state"]
            )
        ),
    }
    # Strict checkpoint reconstruction verifies the train/serve architecture,
    # without preserving this synthetic checkpoint as scientific evidence.
    with tempfile.TemporaryDirectory(
        prefix="nrnsift-checkpoint-", dir="/tmp"
    ) as scratch:
        path = Path(scratch) / "model.pt"
        torch.save(model.state_dict(), path)
        restored = MicroenvironmentPretrainingModel(config, True).to(device)
        restored.load_state_dict(
            torch.load(path, map_location=device, weights_only=True), strict=True
        )
        restored.eval()
        replay_conditions = dict(
            conditions, backbone=restored.operator, sample_count=2, denoising_steps=2
        )
        expected = model.conditional_diffusion.sample(
            **dict(replay_conditions, backbone=model.operator)
        )
        replay = restored.conditional_diffusion.sample(**replay_conditions)
        checks["checkpoint_replay"] = bool(
            torch.equal(expected["trajectory_mean"], replay["trajectory_mean"])
        )
    payload = dict(
        status="engineering_checks_passed"
        if all(checks.values())
        else "engineering_checks_failed",
        scientific_readiness="not_qualified",
        publication_eligible=False,
        data="analytic_fixture_exogenous_occupancy_before_response",
        model_scale="debug_full_architecture",
        job_id=os.environ.get("SLURM_JOB_ID"),
        node=os.environ.get("SLURMD_NODENAME"),
        device=str(device),
        torch_version=torch.__version__,
        trainable_parameters=sum(p.numel() for p in model.parameters()),
        optimization_steps=args.optimization_steps,
        fixed_corruption_loss_before=before,
        fixed_corruption_loss_after=after,
        training_losses=losses,
        stochastic_samples=args.sample_count,
        reverse_steps=args.denoising_steps,
        elapsed_seconds=time.perf_counter() - started,
        checks=checks,
        uncertainty_status=prediction["uncertainty_status"],
        data_lineage_boundary="No research dataset or checkpoint opened; no biological conclusions.",
    )
    write_json(Path(args.output), payload)
    print(json.dumps(payload, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
