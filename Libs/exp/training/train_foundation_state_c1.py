"""Train one hash-bound C1 reduced-core state encoder run.

This entrypoint intentionally owns only the public transcriptomics objective. It
does not create synthetic multimodal pairs or silently substitute a random split.
Every pretraining run obtains the live opaque loader authorization from the
existing C1 preflight path before reading an observation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import platform
import random
import resource
import time
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.exp.training.preflight_foundation_state_c1_run import (  # noqa: E402
    build_authorized_run_preflight,
)
from Libs.models.modules.foundation_state_objectives import (  # noqa: E402
    foundation_state_objective,
)
from Libs.models.networks.foundation_state_encoder import (  # noqa: E402
    NeuronSifterFoundationStateEncoder,
)
from Libs.utils.data.foundation_state_c1_loader import (  # noqa: E402
    C1TranscriptomicsLoader,
    build_c1_loader_context,
    release_c1_run_scope_attestation,
)
from Libs.utils.data.public_foundation_corpus import canonical_sha256  # noqa: E402
from Libs.utils.data.public_foundation_evidence import sha256_file  # noqa: E402
from Libs.utils.data.raw_data_registry import load_config, write_json  # noqa: E402
from Libs.utils.exp_utils import (  # noqa: E402
    C1_RUNTIME_GIT_PATHS,
    compute_file_sha256,
    get_git_provenance,
)


C1_RUN_SCHEMA_VERSION = "neuronsifter-round23-c1-development-run-v1"
C1_CHECKPOINT_SCHEMA_VERSION = "neuronsifter-round23-c1-checkpoint-v1"
C1_ROUNDTRIP_OUTPUT_KEYS = (
    "latent_mean",
    "latent_log_variance",
    "biological_unit_latent_mean",
    "reconstructed_values",
)
C1_ROUNDTRIP_RTOL = 1e-5
C1_ROUNDTRIP_ATOL = 1e-6


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--scale", default="Student", choices=["Debug", "Student", "Large"])
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Optional assertion of the YAML-declared step count; it cannot override it.",
    )
    parser.add_argument("--single-source", default="")
    parser.add_argument("--single-family", default="")
    parser.add_argument("--headline", action="store_true")
    parser.add_argument(
        "--vnext-development",
        action="store_true",
        help="Use the Round-23 development-only step schedule; publication/headline guards remain closed.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("C1 CUDA device was requested but is unavailable")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def _configured_steps(
    args: argparse.Namespace,
    training_contract: Mapping[str, object],
    vnext_contract: Mapping[str, object] | None = None,
) -> int:
    vnext_development = bool(getattr(args, "vnext_development", False))
    if vnext_development and args.headline:
        raise ValueError("--vnext-development cannot be combined with --headline")
    if vnext_development:
        if not isinstance(vnext_contract, Mapping):
            raise ValueError("Round-23 vNext contract is required for --vnext-development")
        schedule_name = "development_steps"
        schedule = vnext_contract["training"][schedule_name]
    else:
        schedule_name = "default_steps" if args.headline else "smoke_steps"
        schedule = training_contract[schedule_name]
    if args.scale not in schedule:
        raise ValueError(
            f"C1 {args.scale} has no YAML-declared {'headline' if args.headline else 'smoke'} step count"
        )
    steps = int(schedule[args.scale])
    if steps <= 0:
        raise ValueError("C1 YAML-declared training steps must be positive")
    if args.steps is not None and int(args.steps) != steps:
        raise ValueError(
            f"C1 --steps={args.steps} does not match YAML-declared {schedule_name}.{args.scale}={steps}"
        )
    return steps


def _precision(
    args: argparse.Namespace, training_contract: Mapping[str, object]
) -> str:
    run_kind = "headline" if args.headline else "development_smoke"
    precision = str(training_contract["precision_by_run_kind"][run_kind])
    if precision not in {"fp32", "bf16"}:
        raise ValueError(f"Unsupported C1 precision: {precision}")
    return precision


def _software_environment() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "platform": platform.platform(),
    }


def _hardware(device: torch.device) -> dict[str, object]:
    payload: dict[str, object] = {
        "device": str(device),
        "host": platform.node(),
        "logical_cpu_count": os.cpu_count(),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        payload.update({
            "gpu_name": properties.name,
            "gpu_total_memory_bytes": int(properties.total_memory),
            "gpu_compute_capability": f"{properties.major}.{properties.minor}",
        })
    return payload


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _model_constructor(
    config: Mapping[str, object],
    loader: C1TranscriptomicsLoader,
    scale: str,
) -> dict[str, object]:
    contract = config["foundation_state_c1"]
    encoder = contract["encoder"]
    return {
        "feature_vocabulary_size": int(loader.vocabulary.feature_vocabulary_size),
        "cell_ontology_size": int(loader.vocabulary.cell_ontology_size),
        "region_ontology_size": int(loader.vocabulary.region_ontology_size),
        "source_class_count": int(loader.vocabulary.source_count),
        "family_class_count": int(loader.vocabulary.family_count),
        "action_feature_dim": int(encoder["input_contract"]["action_feature_dim"]),
        "program_count": 0,
        "program_vocab_status": "not_available",
        "profile": dict(encoder["profiles"][scale]),
        "posterior": dict(encoder["posterior"]),
    }


def _model_from_constructor(
    constructor: Mapping[str, object],
) -> NeuronSifterFoundationStateEncoder:
    return NeuronSifterFoundationStateEncoder(
        feature_vocabulary_size=int(constructor["feature_vocabulary_size"]),
        cell_ontology_size=int(constructor["cell_ontology_size"]),
        region_ontology_size=int(constructor["region_ontology_size"]),
        source_class_count=int(constructor["source_class_count"]),
        family_class_count=int(constructor["family_class_count"]),
        action_feature_dim=int(constructor["action_feature_dim"]),
        program_count=int(constructor["program_count"]),
        program_vocab_status=str(constructor["program_vocab_status"]),
        profile=constructor["profile"],
        posterior=constructor["posterior"],
    )


def _model(
    config: Mapping[str, object], loader: C1TranscriptomicsLoader, scale: str
) -> NeuronSifterFoundationStateEncoder:
    return _model_from_constructor(_model_constructor(config, loader, scale))


def _deterministic_probe_outputs(
    model: NeuronSifterFoundationStateEncoder,
    inputs: Mapping[str, torch.Tensor],
    forward_options: Mapping[str, object],
) -> dict[str, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        outputs = model(
            **inputs,
            use_ontology_context=bool(forward_options["use_ontology_context"]),
            use_action_context=bool(forward_options["use_action_context"]),
            sample_posterior=False,
            nuisance_gradient_scale=float(
                forward_options["nuisance_gradient_scale"]
            ),
        )
    selected = {
        key: outputs[key].detach().cpu().clone()
        for key in C1_ROUNDTRIP_OUTPUT_KEYS
    }
    if not all(torch.isfinite(value).all() for value in selected.values()):
        raise FloatingPointError("C1 deterministic checkpoint probe is non-finite")
    return selected


def _checkpoint_roundtrip(checkpoint_path: Path) -> dict[str, object]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != C1_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("C1 checkpoint schema mismatch during roundtrip")
    restored = _model_from_constructor(payload["model_constructor"])
    restored.load_state_dict(payload["model"], strict=True)
    probe = payload["roundtrip_probe"]
    actual = _deterministic_probe_outputs(
        restored,
        probe["inputs"],
        probe["forward_options"],
    )
    expected = probe["expected_outputs"]
    max_abs_error: dict[str, float] = {}
    matches = True
    for key in C1_ROUNDTRIP_OUTPUT_KEYS:
        if key not in expected or not isinstance(expected[key], torch.Tensor):
            raise ValueError(f"C1 checkpoint probe is missing expected tensor {key}")
        if actual[key].shape != expected[key].shape:
            matches = False
            max_abs_error[key] = float("inf")
            continue
        difference = torch.abs(actual[key] - expected[key])
        max_abs_error[key] = float(difference.max().item()) if difference.numel() else 0.0
        matches = matches and bool(
            torch.allclose(
                actual[key],
                expected[key],
                rtol=C1_ROUNDTRIP_RTOL,
                atol=C1_ROUNDTRIP_ATOL,
            )
        )
    return {
        "passed": bool(matches),
        "output_keys": list(C1_ROUNDTRIP_OUTPUT_KEYS),
        "max_abs_error": max_abs_error,
        "rtol": C1_ROUNDTRIP_RTOL,
        "atol": C1_ROUNDTRIP_ATOL,
    }


def _run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.relative_to(REPO_ROOT.resolve())
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "run_result.json"
    if result_path.exists():
        raise FileExistsError(
            "C1 run output already contains run_result.json; use a new run UID/output directory"
        )
    started = time.time()
    started_at = _utc_now()
    run_uid = os.environ.get("NRNSIFTER_C1_RUN_UID", output_dir.name)
    root_experiment_uid = os.environ.get("NRNSIFTER_C1_ROOT_EXPERIMENT_UID", run_uid)
    authorization = None
    config: Mapping[str, object] | None = None
    preflight: Mapping[str, object] | None = None
    loader: C1TranscriptomicsLoader | None = None
    model: torch.nn.Module | None = None
    losses: list[float] = []
    observations_seen = 0
    biological_units_seen: set[str] = set()
    training_started = False
    result: dict[str, object] = {
        "schema_version": C1_RUN_SCHEMA_VERSION,
        "status": "failed",
        "training_started": False,
        "run_uid": run_uid,
        "root_experiment_uid": root_experiment_uid,
        "lineage_id": "neuronsifter_vnext_dev_v1",
        "domain": "c1",
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
        "protected_test_access": 0,
        "finite_loss": False,
        "finite_uncertainty": False,
        "checkpoint_path": None,
        "checkpoint_sha256": None,
        "checkpoint_roundtrip": False,
        "run_kind": "headline" if args.headline else "development_smoke",
        "headline": bool(args.headline),
        "baseline_id": args.baseline,
        "runtime_profile_id": args.baseline,
        "model_scale": args.scale,
        "seed": args.seed,
        "steps": None,
        "started_at_unix": started,
        "started_at": started_at,
        "host": platform.node(),
        "device": "unresolved",
        "precision": "unresolved",
        "gradient_accumulation_steps": None,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME")
        or os.environ.get("NRNSIFTER_C1_JOB_NAME"),
        "resource_profile": os.environ.get("NRNSIFTER_C1_RESOURCE_PROFILE"),
        "scheduler_stdout_path": os.environ.get(
            "NRNSIFTER_C1_SCHEDULER_STDOUT_PATH", ""
        ),
        "scheduler_stderr_path": os.environ.get(
            "NRNSIFTER_C1_SCHEDULER_STDERR_PATH", ""
        ),
        "software_environment": _software_environment(),
        **get_git_provenance(C1_RUNTIME_GIT_PATHS),
    }
    try:
        config_path = REPO_ROOT / args.config
        config = load_config(config_path)
        contract = config["foundation_state_c1"]
        training_contract = contract["training"]
        steps = _configured_steps(args, training_contract, config.get("neuronsifter_vnext"))
        precision = _precision(args, training_contract)
        accumulation_steps = int(
            training_contract["gradient_accumulation_steps"][args.scale]
        )
        if accumulation_steps <= 0:
            raise ValueError("C1 gradient accumulation steps must be positive")
        result.update({
            "steps": steps,
            "config_path": args.config,
            "config_file_hash": sha256_file(config_path),
            "foundation_state_contract_hash": canonical_sha256(contract),
            "precision": precision,
            "gradient_accumulation_steps": accumulation_steps,
        })
        result["config_hash"] = canonical_sha256(config)
        result["runtime_profile_id"] = args.baseline
        _set_seed(args.seed)
        device = _device(args.device)
        result["device"] = str(device)
        result["hardware"] = _hardware(device)
        runtime_profile = contract["baselines"]["runtime_profiles"][args.baseline]
        if runtime_profile["pretraining_scope"] == "none":
            result.update({
                "status": "not_applicable_pretraining",
                "result_status": "requires_downstream_supervised_probe",
                "training_started": False,
            })
            return result

        if args.headline:
            unified_preflight_path = REPO_ROOT / contract["execution"][
                "unified_preflight_path"
            ]
            unified_preflight = json.loads(
                unified_preflight_path.read_text(encoding="utf-8")
            )
            if (
                unified_preflight.get("status") != "passed"
                or unified_preflight.get("headline_training_allowed") is not True
                or unified_preflight.get("blockers")
                or unified_preflight.get("headline_requirements")
            ):
                raise RuntimeError(
                    "C1 headline training is blocked by the unified preflight"
                )

        preflight, authorization = build_authorized_run_preflight(
            config,
            baseline_id=args.baseline,
            split_uid=args.split,
            model_scale=args.scale,
            seed=args.seed,
            single_source=args.single_source,
            single_family=args.single_family,
            headline=args.headline,
        )
        result.update({
            "corpus_hash": preflight["corpus_hash"],
            "split_hash": preflight["split_manifest_hash"],
            "run_scope_binding_hash": preflight["run_scope_binding_hash"],
            "active_objectives": list(preflight["run_scope_binding"]["active_objectives"]),
        })
        context = build_c1_loader_context(REPO_ROOT, config)
        binding = preflight["run_scope_binding"]
        role_path = REPO_ROOT / preflight["train_role_binding"]["path"]
        loader = C1TranscriptomicsLoader(
            REPO_ROOT,
            config,
            role_binding_path=role_path,
            expected_role="train",
            batch_size=int(binding["batch_size"]),
            profile=args.scale,
            seed=args.seed,
            sampling_policy=str(binding["sampling_policy"]),
            max_observations_per_unit=int(binding["max_observations_per_biological_unit"]),
            objective_set=str(binding["objective_set"]),
            source_id=binding["source_id"],
            family_uid=binding["family_uid"],
            run_scope_attestation=authorization,
            loader_context=context,
        )
        device_obj = device
        model_constructor = _model_constructor(config, loader, args.scale)
        model = _model_from_constructor(model_constructor).to(device_obj)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(training_contract["learning_rate"])
        )
        loss_weights = contract["encoder"]["loss_weights"]
        active_objectives = tuple(str(value) for value in binding["active_objectives"])
        model.train()
        autocast_enabled = precision == "bf16"
        if autocast_enabled and device_obj.type != "cuda":
            raise RuntimeError("C1 bf16 headline precision requires CUDA")
        for step in range(steps):
            batch = loader.batch(epoch=0, batch_index=step)
            training_started = True
            observations_seen += int(batch.observation_unit_index.numel())
            biological_units_seen.update(batch.biological_unit_uids)
            tensors = {
                "feature_ids": batch.feature_ids.to(device_obj),
                "feature_values": batch.feature_values.to(device_obj),
                "observed_mask": batch.observed_mask.to(device_obj),
                "target_mask": batch.target_mask.to(device_obj),
                "cell_ontology_ids": batch.cell_ontology_ids.to(device_obj),
                "region_ontology_ids": batch.region_ontology_ids.to(device_obj),
                "action_features": batch.action_features.to(device_obj),
                "action_available": batch.action_available.to(device_obj),
                "observation_unit_index": batch.observation_unit_index.to(device_obj),
            }
            with torch.autocast(
                device_type=device_obj.type,
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                outputs = model(
                    **tensors,
                    **model.runtime_flags(active_objectives),
                    sample_posterior=True,
                    nuisance_gradient_scale=float(contract["encoder"]["nuisance_gradient_scale"]),
                )
                objective = foundation_state_objective(
                    outputs,
                    tensors["feature_values"],
                    tensors["target_mask"],
                    tensors["observation_unit_index"],
                    batch.biological_unit_weights.to(device_obj),
                    active_objectives=active_objectives,
                    loss_weights=loss_weights,
                    semantic_group_ids=batch.semantic_group_ids.to(device_obj),
                    source_ids=batch.source_ids.to(device_obj),
                    family_ids=batch.family_ids.to(device_obj),
                    contrastive_temperature=float(contract["encoder"]["contrastive_temperature"]),
                )
            optimizer.zero_grad(set_to_none=True)
            objective.total.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training_contract["gradient_clip_norm"])
            )
            optimizer.step()
            value = float(objective.total.detach().cpu())
            if not np.isfinite(value):
                raise FloatingPointError("C1 objective became non-finite")
            losses.append(value)
            print(json.dumps({"step": step + 1, "loss": value}, sort_keys=True), flush=True)
        model_profile = dict(contract["encoder"]["profiles"][args.scale])
        checkpoint_path = output_dir / "last.pt"
        checkpoint_relative_path = _repo_relative(checkpoint_path)
        model_state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        probe_inputs = {
            name: value.detach().cpu().clone()
            for name, value in tensors.items()
        }
        forward_options: dict[str, object] = {
            **model.runtime_flags(active_objectives),
            "nuisance_gradient_scale": float(
                contract["encoder"]["nuisance_gradient_scale"]
            ),
        }
        probe_model = _model_from_constructor(model_constructor)
        probe_model.load_state_dict(model_state, strict=True)
        expected_outputs = _deterministic_probe_outputs(
            probe_model,
            probe_inputs,
            forward_options,
        )
        finite_uncertainty = bool(
            torch.isfinite(expected_outputs["latent_log_variance"]).all()
            and torch.isfinite(
                torch.exp(0.5 * expected_outputs["latent_log_variance"])
            ).all()
        )
        if not finite_uncertainty:
            raise FloatingPointError("C1 checkpoint probe uncertainty is non-finite")
        checkpoint_payload = {
            "schema_version": C1_CHECKPOINT_SCHEMA_VERSION,
            "lineage_id": "neuronsifter_vnext_dev_v1",
            "domain": "c1",
            "evidence_role": "development_only",
            "publication_claim_eligible": False,
            "protected_test_access": 0,
            "checkpoint_path": checkpoint_relative_path,
            "model": model_state,
            "optimizer": optimizer.state_dict(),
            "baseline_id": args.baseline,
            "model_scale": args.scale,
            "seed": args.seed,
            "corpus_hash": preflight["corpus_hash"],
            "split_hash": preflight["split_manifest_hash"],
            "run_scope_binding_hash": preflight["run_scope_binding_hash"],
            "selection_metric": "train_objective_smoke_only",
            "selection_status": "not_available_smoke_only",
            "config_hash": result["config_hash"],
            "foundation_state_contract_hash": result["foundation_state_contract_hash"],
            "model_profile": model_profile,
            "model_constructor": model_constructor,
            "finite_uncertainty": finite_uncertainty,
            "roundtrip_probe": {
                "sample_posterior": False,
                "inputs": probe_inputs,
                "forward_options": forward_options,
                "expected_outputs": expected_outputs,
                "output_keys": list(C1_ROUNDTRIP_OUTPUT_KEYS),
            },
        }
        if checkpoint_path.exists():
            raise FileExistsError(
                "C1 run output already contains last.pt; use a new run UID/output directory"
            )
        torch.save(checkpoint_payload, checkpoint_path)
        checkpoint_hash = compute_file_sha256(checkpoint_path)
        roundtrip = _checkpoint_roundtrip(checkpoint_path)
        if roundtrip["passed"] is not True:
            raise RuntimeError("C1 checkpoint roundtrip changed deterministic predictions")
        checkpoint_pointer = REPO_ROOT / (
            f"Output/checkpoints/round23/c1_{args.scale.lower()}_seed{int(args.seed)}.json"
        )
        if checkpoint_pointer.exists():
            raise FileExistsError(
                "C1 checkpoint pointer already exists; preserve it before rerunning this scale and seed"
            )
        write_json(
            checkpoint_pointer,
            {
                "schema_version": "neuronsifter-round23-checkpoint-pointer-v1",
                "checkpoint_path": checkpoint_relative_path,
                "checkpoint_sha256": checkpoint_hash,
                "run_uid": run_uid,
                "model_scale": args.scale,
                "seed": int(args.seed),
                "evidence_role": "development_only",
                "publication_claim_eligible": False,
                "protected_test_access": 0,
            },
        )
        activated_parameter_count = int(
            sum(
                parameter.numel()
                for name, parameter in model.named_parameters()
                if not (
                    "program_head" in name
                    or (
                        "source_nuisance_head" in name
                        and "source_family_nuisance_regularization"
                        not in active_objectives
                    )
                    or (
                        "family_nuisance_head" in name
                        and "source_family_nuisance_regularization"
                        not in active_objectives
                    )
                )
            )
        )
        result.update({
            "status": "passed",
            "training_started": training_started,
            "result_status": "trained",
            "losses": losses,
            "mean_loss": float(np.mean(losses)),
            "finite_loss": bool(losses) and all(np.isfinite(value) for value in losses),
            "finite_uncertainty": finite_uncertainty,
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "trainable_parameter_count": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
            "activated_parameter_count": activated_parameter_count,
            "model_profile": model_profile,
            "embedding_width": int(model_profile["latent_dim"]),
            "depth": int(model_profile["transformer_layers"]),
            "effective_biological_units_seen": int(steps * loader.batch_size),
            "unique_biological_units_seen": len(biological_units_seen),
            "effective_observations_seen": observations_seen,
            "biological_units_per_batch": int(loader.batch_size),
            "batch_construction": str(binding["sampling_policy"]),
            "checkpoint_hash": checkpoint_hash,
            "checkpoint_path": checkpoint_relative_path,
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_roundtrip": True,
            "checkpoint_roundtrip_output_keys": roundtrip["output_keys"],
            "checkpoint_roundtrip_max_abs_error": roundtrip["max_abs_error"],
            "checkpoint_roundtrip_rtol": roundtrip["rtol"],
            "checkpoint_roundtrip_atol": roundtrip["atol"],
            "checkpoint_pointer": _repo_relative(checkpoint_pointer),
            "last_checkpoint_hash": checkpoint_hash,
            "best_validation_checkpoint_hash": "",
            "checkpoint_selection_metric": "train_objective_smoke_only",
            "checkpoint_selection_status": "not_available_smoke_only",
        })
        return result
    except Exception as exc:
        result.update({
            "status": "failed",
            "training_started": training_started,
            "result_status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        raise
    finally:
        if authorization is not None:
            release_c1_run_scope_attestation(authorization)
        result["training_started"] = bool(result.get("training_started") or training_started)
        result["ended_at_unix"] = time.time()
        result["ended_at"] = _utc_now()
        result["elapsed_seconds"] = result["ended_at_unix"] - started
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result["resource_usage"] = {
            "user_cpu_seconds": float(usage.ru_utime),
            "system_cpu_seconds": float(usage.ru_stime),
            "max_rss_kib": int(usage.ru_maxrss),
            "cuda_max_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
            ),
            "cuda_max_memory_reserved_bytes": (
                int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0
            ),
        }
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


def main() -> None:
    result = _run(parse_args())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
