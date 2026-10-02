"""Build development-only Round-23 C1 and C2-S diagnostic reports."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.exp.training.train_foundation_state_c1 import _model as build_c1_model  # noqa: E402
from Libs.exp.training.train_restricted_c2_vnext import _sha256_file  # noqa: E402
from Libs.utils.data.foundation_state_c1_loader import (  # noqa: E402
    C1TranscriptomicsLoader,
    build_c1_loader_context,
)
from Libs.utils.data.raw_data_registry import load_config, write_tsv  # noqa: E402
from Libs.utils.evaluation.foundation_state_c1_metrics import (  # noqa: E402
    chance_normalized_balanced_accuracy,
)


def _repo_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def _c1_row(
    *,
    diagnostic_id: str,
    metric_id: str,
    value: float | None,
    unit: str,
    status: str,
    metric_role: str,
    checkpoint_path: str,
    checkpoint_sha256: str,
    split_uid: str,
    batch_count: int,
    biological_unit_count: int,
    observation_count: int,
) -> dict[str, Any]:
    return {
        "diagnostic_id": diagnostic_id,
        "metric_id": metric_id,
        "value": "" if value is None else float(value),
        "unit": unit,
        "status": status,
        "metric_role": metric_role,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "model_scale": "Large",
        "seed": 3407,
        "split_uid": split_uid,
        "split_role": "validation",
        "batch_count": batch_count,
        "biological_unit_count": biological_unit_count,
        "observation_count": observation_count,
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
        "protected_test_access": 0,
    }


def build_c1(
    *,
    checkpoint_path: Path,
    checkpoint_identity_path: Path,
    device: str,
) -> list[dict[str, Any]]:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    diagnostics = config["neuronsifter_vnext"]["development_diagnostics"]["c1"]
    canonical_checkpoint_path = _repo_path(str(diagnostics["checkpoint_path"]))
    if checkpoint_identity_path.resolve() != canonical_checkpoint_path:
        raise ValueError("Round-23 C1 diagnostic checkpoint identity drift")
    run_result_path = canonical_checkpoint_path.parent / "run_result.json"
    run_result = json.loads(run_result_path.read_text(encoding="utf-8"))
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    if checkpoint_sha256 != str(run_result["checkpoint_hash"]):
        raise ValueError("Round-23 C1 diagnostic checkpoint hash drift")
    if run_result.get("status") != "passed" or run_result.get("model_scale") != "Large":
        raise ValueError("Round-23 C1 diagnostic checkpoint is not a passed Large run")
    role_binding_path = _repo_path(str(diagnostics["role_binding_path"]))
    loader_context = build_c1_loader_context(REPO_ROOT, config)
    loader = C1TranscriptomicsLoader(
        REPO_ROOT,
        config,
        role_binding_path=role_binding_path,
        expected_role="validation",
        batch_size=int(diagnostics["batch_size"]),
        profile="Large",
        seed=int(run_result["seed"]),
        sampling_policy=str(config["foundation_state_c1"]["loader"]["sampling_policy"]),
        max_observations_per_unit=int(
            config["foundation_state_c1"]["loader"][
                "max_observations_per_biological_unit"
            ]
        ),
        objective_set="masked_only",
        run_scope_attestation=None,
        loader_context=loader_context,
    )
    device_object = torch.device(device)
    if device_object.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for C1 diagnostics but is unavailable")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_c1_model(config, loader, "Large").to(device_object)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    active_objectives = tuple(str(value) for value in run_result["active_objectives"])
    runtime_flags = model.runtime_flags(active_objectives)
    nuisance_scale = float(
        config["foundation_state_c1"]["encoder"]["nuisance_gradient_scale"]
    )
    posterior_sample_count = int(diagnostics["posterior_sample_count"])
    interval_level = float(diagnostics["calibration_interval_level"])
    if not math.isclose(interval_level, 0.9):
        raise ValueError("Round-23 C1 calibration interval is frozen at 0.9")
    z_value = 1.6448536269514722

    residuals: list[np.ndarray] = []
    absolute_residuals: list[np.ndarray] = []
    interval_hits: list[np.ndarray] = []
    source_predictions: list[np.ndarray] = []
    source_labels: list[np.ndarray] = []
    source_squared_errors: dict[int, list[float]] = {}
    unit_uids: set[str] = set()
    observation_count = 0
    batch_count = int(diagnostics["batch_count"])
    with torch.no_grad():
        for batch_index in range(batch_count):
            batch = loader.batch(epoch=0, batch_index=batch_index)
            unit_uids.update(batch.biological_unit_uids)
            observation_count += int(batch.feature_ids.shape[0])
            tensors = {
                key: value.to(device_object) for key, value in batch.model_inputs().items()
            }
            deterministic = model(
                **tensors,
                **runtime_flags,
                sample_posterior=False,
                nuisance_gradient_scale=nuisance_scale,
            )
            target_mask = tensors["target_mask"]
            truth = tensors["feature_values"][target_mask]
            prediction = deterministic["reconstructed_values"][target_mask]
            residual = (truth - prediction).detach().cpu().numpy().astype(np.float64)
            residuals.append(np.square(residual))
            absolute_residuals.append(np.abs(residual))
            source_predictions.append(
                deterministic["source_diagnostic_logits"].argmax(dim=1).cpu().numpy()
            )
            source_labels.append(batch.source_ids.numpy())
            per_observation_squared = (
                (tensors["feature_values"] - deterministic["reconstructed_values"])
                .square()
                .masked_fill(~target_mask, 0.0)
                .sum(dim=1)
                / target_mask.sum(dim=1).clamp_min(1)
            ).cpu().numpy()
            for source_id, squared_error in zip(
                batch.source_ids.tolist(), per_observation_squared.tolist(), strict=True
            ):
                source_squared_errors.setdefault(int(source_id), []).append(
                    float(squared_error)
                )
            posterior_draws = []
            for sample_index in range(posterior_sample_count):
                torch.manual_seed(3407 + batch_index * 100 + sample_index)
                posterior_draws.append(
                    model(
                        **tensors,
                        **runtime_flags,
                        sample_posterior=True,
                        nuisance_gradient_scale=nuisance_scale,
                    )["reconstructed_values"][target_mask]
                )
            stacked = torch.stack(posterior_draws, dim=0)
            posterior_mean = stacked.mean(dim=0)
            posterior_std = stacked.std(dim=0, unbiased=False).clamp_min(1e-6)
            interval_hits.append(
                (
                    torch.abs(truth - posterior_mean) <= z_value * posterior_std
                )
                .cpu()
                .numpy()
            )

    squared = np.concatenate(residuals)
    absolute = np.concatenate(absolute_residuals)
    coverage = float(np.mean(np.concatenate(interval_hits)))
    source_prediction = np.concatenate(source_predictions)
    source_label = np.concatenate(source_labels)
    observed_sources = sorted(set(int(value) for value in source_label.tolist()))
    recalls = [
        float(np.mean(source_prediction[source_label == source] == source))
        for source in observed_sources
    ]
    balanced_accuracy = float(np.mean(recalls))
    chance_normalized = chance_normalized_balanced_accuracy(
        balanced_accuracy, loader.vocabulary.source_count
    )
    source_rmse_values = [
        math.sqrt(float(np.mean(values))) for values in source_squared_errors.values()
    ]
    checkpoint_identity = checkpoint_identity_path.relative_to(REPO_ROOT).as_posix()
    common = {
        "checkpoint_path": checkpoint_identity,
        "checkpoint_sha256": checkpoint_sha256,
        "split_uid": str(diagnostics["split_uid"]),
        "batch_count": batch_count,
        "biological_unit_count": len(unit_uids),
        "observation_count": observation_count,
    }
    rows = [
        _c1_row(
            diagnostic_id="masked_state_reconstruction",
            metric_id="masked_target_rmse",
            value=math.sqrt(float(np.mean(squared))),
            unit="normalized_expression",
            status="measured",
            metric_role="direct_development_diagnostic",
            **common,
        ),
        _c1_row(
            diagnostic_id="masked_state_reconstruction",
            metric_id="masked_target_mae",
            value=float(np.mean(absolute)),
            unit="normalized_expression",
            status="measured",
            metric_role="direct_development_diagnostic",
            **common,
        ),
        _c1_row(
            diagnostic_id="missing_modality",
            metric_id="missing_modality_robustness",
            value=None,
            unit="not_available",
            status=str(diagnostics["missing_modality_status"]),
            metric_role="declared_unavailable_not_imputed",
            **common,
        ),
        _c1_row(
            diagnostic_id="source_predictability",
            metric_id="source_balanced_accuracy",
            value=balanced_accuracy,
            unit="fraction",
            status="measured",
            metric_role="shortcut_diagnostic",
            **common,
        ),
        _c1_row(
            diagnostic_id="source_predictability",
            metric_id="source_chance_normalized_balanced_accuracy",
            value=chance_normalized,
            unit="fraction",
            status="measured",
            metric_role="shortcut_diagnostic",
            **common,
        ),
        _c1_row(
            diagnostic_id="cross_source_development_transfer",
            metric_id="source_stratified_masked_rmse_macro_proxy",
            value=float(np.mean(source_rmse_values)),
            unit="normalized_expression",
            status=str(diagnostics["cross_source_transfer_status"]),
            metric_role="proxy_not_cross_source_holdout",
            **common,
        ),
        _c1_row(
            diagnostic_id="uncertainty_calibration",
            metric_id="posterior_reconstruction_interval_coverage",
            value=coverage,
            unit="fraction",
            status="measured",
            metric_role="development_calibration",
            **common,
        ),
        _c1_row(
            diagnostic_id="uncertainty_calibration",
            metric_id="posterior_reconstruction_calibration_absolute_error",
            value=abs(interval_level - coverage),
            unit="fraction",
            status="measured",
            metric_role="development_calibration",
            **common,
        ),
    ]
    write_tsv(
        REPO_ROOT / "Output/paper_reports/round23_dev_c1_diagnostics.tsv",
        list(rows[0]),
        rows,
    )
    return rows


def build_c2s() -> list[dict[str, Any]]:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    diagnostics = config["neuronsifter_vnext"]["development_diagnostics"]["c2s"]
    metric_fields = {
        "trajectory_error_rmse_raw": "trajectory_rmse_raw",
        "long_horizon_drift_rmse_raw": "long_horizon_drift_rmse_raw",
        "action_sensitivity_mean_abs_raw": "action_sensitivity_mean_abs_raw",
        "structured_action_contrast_rmse_raw": "structured_action_contrast_rmse_raw",
        "crps_normalized": "crps_normalized",
        "nll_normalized": "nll_normalized",
        "constraint_negative_fraction_proxy": "constraint_negative_fraction",
        "epistemic_variance_mean": "epistemic_variance_mean",
        "aleatoric_variance_mean": "aleatoric_variance_mean",
        "calibration_interval_coverage": "calibration_interval_coverage",
        "calibration_absolute_error": "calibration_absolute_error",
    }
    rows: list[dict[str, Any]] = []
    seeds = [int(value) for value in diagnostics["selected_seeds"]]
    for seed in seeds:
        result_path = (
            REPO_ROOT / f"Output/runs/round23/c2s/large_seed{seed}/run_result.json"
        )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            result.get("status") != "passed"
            or result.get("model_scale") != diagnostics["selected_model_scale"]
            or int(result.get("seed", -1)) != seed
            or result.get("checkpoint_roundtrip") is not True
            or int(result.get("protected_test_access", -1)) != 0
        ):
            raise ValueError(f"Round-23 C2-S diagnostic run is not admissible: {seed}")
        row: dict[str, Any] = {
            "row_type": "seed",
            "run_uid": result["run_uid"],
            "model_scale": result["model_scale"],
            "seed": seed,
            "checkpoint_path": result["checkpoint_path"],
            "checkpoint_sha256": result["checkpoint_sha256"],
            "constraint_metric_semantics": diagnostics[
                "constraint_metric_semantics"
            ],
        }
        for output_field, input_field in metric_fields.items():
            value = float(result["metrics"][input_field])
            if not math.isfinite(value):
                raise ValueError(f"Round-23 C2-S diagnostic is non-finite: {input_field}")
            row[output_field] = value
        row.update(
            {
                "finite_trajectory": bool(result["finite_trajectory"]),
                "finite_uncertainty": bool(result["finite_uncertainty"]),
                "evidence_role": "development_only",
                "publication_claim_eligible": False,
                "protected_test_access": 0,
            }
        )
        rows.append(row)
    aggregate: dict[str, Any] = {
        "row_type": "aggregate_mean",
        "run_uid": "large_seed_aggregate",
        "model_scale": diagnostics["selected_model_scale"],
        "seed": "aggregate",
        "checkpoint_path": "multiple_hash_bound_checkpoints",
        "checkpoint_sha256": "multiple",
        "constraint_metric_semantics": diagnostics["constraint_metric_semantics"],
    }
    for output_field in metric_fields:
        aggregate[output_field] = float(
            np.mean([float(row[output_field]) for row in rows])
        )
    aggregate.update(
        {
            "finite_trajectory": True,
            "finite_uncertainty": True,
            "evidence_role": "development_only",
            "publication_claim_eligible": False,
            "protected_test_access": 0,
        }
    )
    rows.append(aggregate)
    write_tsv(
        REPO_ROOT / "Output/paper_reports/round23_dev_c2s_diagnostics.tsv",
        list(rows[0]),
        rows,
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("c1", "c2s"), required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--checkpoint-identity", default="")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    arguments = parser.parse_args()
    if arguments.domain == "c1":
        if not arguments.checkpoint or not arguments.checkpoint_identity:
            raise ValueError("C1 diagnostics require staged and canonical checkpoint paths")
        rows = build_c1(
            checkpoint_path=Path(arguments.checkpoint).resolve(),
            checkpoint_identity_path=_repo_path(arguments.checkpoint_identity),
            device=arguments.device,
        )
    else:
        rows = build_c2s()
    print(json.dumps({"domain": arguments.domain, "row_count": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
