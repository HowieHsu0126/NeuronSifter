"""Predictive scores in the fixed native training coordinate system."""
from __future__ import annotations

import json
import math
from collections import Counter
from typing import Any

import torch

from Libs.utils.model.screening_model_config import native_direct_mean_config

from Libs.utils.data.native_field_normalization import NativeFieldNormalization
from Libs.utils.data.native_observable_profile import quantity_support

METRIC_NAMES = ("crps", "normalized_rmse", "predictive_coverage", "predictive_calibration_error")
UNDEFINED_STATUSES = {"fully_masked", "insufficient_draws"}


def metric_policy(config: dict) -> dict:
    policy = dict(config["paper_grade_full_suite"]["native_rollout_evaluation"])
    policy.setdefault("predictor_kind", "native_diffusion")
    if policy["predictor_kind"] not in {"native_diffusion", "initial_state_persistence", "transolver_direct_mean", "transolver_fitted_residual", "linear_dynamics_direct_mean"}:
        raise ValueError("Unsupported native evaluation predictor_kind")
    backbone_kind = config.get("screening_pipeline", {}).get("screening_benchmark", {}).get("model_architecture", {}).get("rollout_operator", {}).get("transition_backbone", {}).get("kind")
    if policy["predictor_kind"] == "linear_dynamics_direct_mean" or backbone_kind == "native_linear_dynamics":
        if policy["predictor_kind"] != "linear_dynamics_direct_mean" or backbone_kind != "native_linear_dynamics":
            raise ValueError("Linear predictor/evaluation selector mismatch; residual authority is unbound")
        native_direct_mean_config(config)
    expected = {"crps_estimator": "off_diagonal_finite_ensemble",
                "quantile_interpolation": "linear", "cell_reduction": "equal_valid_cells",
                "episode_reduction": "equal_episode", "episode_field": "sample_uid",
                "group_field": "family_uid"}
    for key, value in expected.items():
        if policy.get(key) != value:
            raise ValueError(f"Unsupported native rollout metric policy: {key}")
    if policy.get("group_kind") not in {"action_family", "scenario_family"}:
        raise ValueError("Native rollout scoring requires declared family grouping semantics")
    return policy


def metric_protocol(policy: dict, quantile_levels: list[float], normalization_id: str,
                    *, draw_semantics: str = "emitted_whole_particle_mixture_draws") -> dict:
    levels = list(quantile_levels)
    if (len(levels) != 3 or not all(isinstance(q, (int, float)) and math.isfinite(q) for q in levels)
            or not 0 < levels[0] < levels[1] < levels[2] < 1
            or levels[1] != .5 or not math.isclose(levels[0] + levels[2], 1.0)):
        raise ValueError("Native predictive coverage requires lower/median/upper central quantiles")
    if draw_semantics not in {"emitted_whole_particle_mixture_draws", "deterministic_point_mass", "mean_only"}:
        raise ValueError("Unsupported native predictive law")
    if draw_semantics == "deterministic_point_mass" and policy["predictor_kind"] != "initial_state_persistence":
        raise ValueError("Analytic point-mass scoring is restricted to the persistence law")
    protocol = {key: policy[key] for key in ("crps_estimator", "quantile_interpolation",
            "cell_reduction", "episode_reduction", "episode_field", "group_field", "group_kind")} | {
        "quantile_levels": levels, "nominal_coverage": levels[-1] - levels[0],
        "normalization_id": normalization_id, "coordinates": "fixed_training_normalized",
        "draw_semantics": draw_semantics,
        "convention_status": "prospective_implementation_choice"}
    if draw_semantics == "deterministic_point_mass":
        protocol.update(crps_estimator="analytic_point_mass_absolute_error",
                        quantile_interpolation="analytic_point_mass")
    if draw_semantics == "mean_only":
        if policy["predictor_kind"] not in {"transolver_direct_mean", "linear_dynamics_direct_mean"}:
            raise ValueError("Mean-only scoring requires a native direct predictor")
        protocol.update(prediction_kind="mean_only", residual_likelihood_status="unbound",
                        crps_estimator="unavailable_residual_likelihood_unbound",
                        quantile_interpolation="unavailable_residual_likelihood_unbound",
                        missing_metric_reason="residual_likelihood_unbound")
    return protocol


def _episode_fields(target: torch.Tensor, mask: torch.Tensor, identity: dict,
                    normalization: NativeFieldNormalization, grid: dict, draw_count: int) -> tuple[dict, torch.Tensor]:
    if not normalization.active:
        raise ValueError("Native predictive scoring requires the fixed R60 training normalization")
    normalization.assert_binding(grid)
    normalization.assert_domain(grid)
    tail = tuple(len(grid[key]) for key in ("time_hours", "region_id", "compartment_id", "species_id"))
    if target.ndim != 5 or tuple(target.shape[1:]) != tail or target.shape[0] < 1:
        raise ValueError("Native predictive target requires exact action/time/grid axes")
    if mask.dtype != torch.bool or mask.shape != target.shape:
        raise ValueError("Native predictive validity mask must be explicit bool with exact target shape")
    if len(identity["regimen_uids"]) != target.shape[0]:
        raise ValueError("Native predictive action identities do not match the action axis")
    valid = (mask & torch.as_tensor(quantity_support(grid), device=mask.device)).flatten()
    row = dict(identity, draw_count=draw_count, total_cell_count=target.numel(),
               valid_cell_count=int(valid.sum()), metrics={name: None for name in METRIC_NAMES})
    return row, valid


def score_native_episode(
    samples: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, identity: dict,
    normalization: NativeFieldNormalization, grid: dict, protocol: dict,
) -> dict:
    """Score actual [draw,action,time,region,compartment,species] samples."""
    if protocol["draw_semantics"] != "emitted_whole_particle_mixture_draws":
        raise ValueError("Monte Carlo scoring requires an emitted predictive-draw law")
    if samples.ndim != 6 or samples.shape[1:] != target.shape or samples.shape[0] < 1:
        raise ValueError("Native predictive samples/target require exact action/time/grid axes")
    row, valid = _episode_fields(target, mask, identity, normalization, grid, int(samples.shape[0]))
    if not bool(valid.any()):
        return dict(row, status="fully_masked")
    x = normalization.normalize(samples.double(), torch.as_tensor(quantity_support(grid), device=samples.device)).reshape(samples.shape[0], -1)[:, valid]
    y = normalization.normalize(target.double(), torch.as_tensor(quantity_support(grid), device=target.device)).flatten()[valid]
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("Native predictive samples/targets must be finite on every valid cell")
    if x.shape[0] < 2:
        return dict(row, status="insufficient_draws")
    ordered = x.sort(dim=0).values
    d = x.shape[0]
    coefficients = (2 * torch.arange(d, dtype=x.dtype, device=x.device) - d + 1).unsqueeze(1)
    # Sum unordered |x_i-x_j| pairs without a draw-by-draw field tensor.
    pair_sum = (coefficients * ordered).sum(0)
    crps = ((x - y).abs().mean(0) - pair_sum / (d * (d - 1))).mean()
    rmse = ((x.mean(0) - y).square().mean()).sqrt()
    quantiles = torch.quantile(x, x.new_tensor(protocol["quantile_levels"]), dim=0,
                              interpolation=protocol["quantile_interpolation"])
    coverage = ((y >= quantiles[0]) & (y <= quantiles[-1])).double().mean()
    row["metrics"].update(crps=float(crps), normalized_rmse=float(rmse),
                          predictive_coverage=float(coverage))
    # Calibration error is defined after equal-episode coverage aggregation.
    return dict(row, status="defined")


def score_point_mass_episode(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, identity: dict,
    normalization: NativeFieldNormalization, grid: dict, protocol: dict,
) -> dict:
    """Analytic deterministic-law scores, with no Monte Carlo draw axis."""
    if (protocol["draw_semantics"] != "deterministic_point_mass"
            or protocol["crps_estimator"] != "analytic_point_mass_absolute_error"):
        raise ValueError("Point-mass scoring requires the declared analytic law")
    if prediction.shape != target.shape:
        raise ValueError("Point-mass prediction/target axes mismatch")
    row, valid = _episode_fields(target, mask, identity, normalization, grid, 0)
    if not bool(valid.any()):
        return dict(row, status="fully_masked")
    x = normalization.normalize(prediction.double(), torch.as_tensor(quantity_support(grid), device=prediction.device)).flatten()[valid]
    y = normalization.normalize(target.double(), torch.as_tensor(quantity_support(grid), device=target.device)).flatten()[valid]
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("Point-mass prediction/target must be finite on valid cells")
    row["metrics"].update(crps=float((x-y).abs().mean()),
                          normalized_rmse=float((x-y).square().mean().sqrt()),
                          predictive_coverage=float((y == x).double().mean()))
    return dict(row, status="defined")


def score_mean_only_episode(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                            identity: dict, normalization: NativeFieldNormalization, grid: dict, protocol: dict) -> dict:
    """A conditional mean estimates one moment; it does not define a field law."""
    if protocol["draw_semantics"] != "mean_only" or protocol.get("residual_likelihood_status") != "unbound":
        raise ValueError("Mean-only scoring requires an explicit unbound residual")
    if prediction.shape != target.shape:
        raise ValueError("Mean-only prediction/target axes mismatch")
    row, valid = _episode_fields(target, mask, identity, normalization, grid, 0)
    row.update(prediction_kind="mean_only", residual_likelihood_status="unbound",
               missing_metric_reason="residual_likelihood_unbound")
    if not bool(valid.any()):
        return dict(row, status="fully_masked")
    x = normalization.normalize(prediction.double(), torch.as_tensor(quantity_support(grid), device=prediction.device)).flatten()[valid]
    y = normalization.normalize(target.double(), torch.as_tensor(quantity_support(grid), device=target.device)).flatten()[valid]
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("Mean-only prediction/target must be finite on valid cells")
    row["metrics"]["normalized_rmse"] = float((x-y).square().mean().sqrt())
    return dict(row, status="residual_unbound")


def failed_native_episode(identity: dict, reason: str, *, draw_count: int = 0) -> dict:
    return dict(identity, status="failed", reason=reason, draw_count=draw_count,
                total_cell_count=None, valid_cell_count=None,
                metrics={name: None for name in METRIC_NAMES})


def build_native_metric_payload(rows: list[dict], protocol: dict, provenance: dict, config: dict) -> dict:
    if not rows or len({row["episode_uid"] for row in rows}) != len(rows):
        raise ValueError("Native metric episodes must be nonempty with unique declared identities")
    statuses = Counter(row["status"] for row in rows)
    if not set(statuses) <= {"defined", "failed", "residual_unbound", *UNDEFINED_STATUSES}:
        raise ValueError("Unknown native episode scoring status")
    defined = [row for row in rows if row["status"] == "defined"]
    n = len(defined)
    metrics = {}
    for name in METRIC_NAMES[:-1]:
        values = [row["metrics"][name] for row in rows if row["metrics"][name] is not None]
        metrics[name] = {"mean":sum(values)/len(values) if values else None, "count":len(values)}
    coverage = metrics["predictive_coverage"]
    metrics["predictive_calibration_error"] = {
        "mean":abs(coverage["mean"]-protocol["nominal_coverage"]) if coverage["count"] else None,
        "count":coverage["count"]}
    groups = {row["family_uid"] for row in rows}
    draws = {row["draw_count"] for row in rows
             if row["draw_count"] > 0 or protocol["draw_semantics"] in {"deterministic_point_mass", "mean_only"}}
    payload = {
        "schema_version": config["schemas"]["native_rollout_metrics"]["schema_version"],
        "status": "complete" if n == len(rows) and protocol["draw_semantics"] != "mean_only" else "incomplete",
        "source_status": "native_development_prediction",
        "metrics": metrics, "episodes": rows, "protocol": protocol, "provenance": provenance,
        "counts": {"episode_count": len(rows), "defined_episode_count": n,
                   "undefined_episode_count": sum(statuses[key] for key in UNDEFINED_STATUSES),
                   **({"partially_defined_episode_count": statuses["residual_unbound"]} if protocol["draw_semantics"] == "mean_only" else {}),
                   "failed_episode_count": statuses["failed"], "group_count": len(groups),
                   "defined_group_count": len({row["family_uid"] for row in defined}),
                   "valid_cell_count": sum(row["valid_cell_count"] or 0 for row in rows),
                   "realized_draw_count": next(iter(draws)) if len(draws) == 1 else None},
    }
    validate_native_metric_payload(payload, config)
    return payload


def validate_native_metric_payload(payload: dict[str, Any], config: dict) -> None:
    schema = config["schemas"]["native_rollout_metrics"]
    if payload.get("schema_version") != schema["schema_version"]:
        raise ValueError("Native metric schema version mismatch")
    if not set(schema["required_fields"]) <= set(payload):
        raise ValueError("Missing native metric payload fields")
    rows, counts = payload["episodes"], payload["counts"]
    protocol = payload["protocol"]
    if not set(schema["protocol_required_fields"]) <= set(protocol):
        raise ValueError("Missing predictive law/estimator protocol")
    if protocol["draw_semantics"] not in schema["draw_semantics"]:
        raise ValueError("Unknown predictive law")
    if protocol["draw_semantics"] == "deterministic_point_mass":
        if (protocol["crps_estimator"] != "analytic_point_mass_absolute_error"
                or protocol["quantile_interpolation"] != "analytic_point_mass"
                or any(row["draw_count"] != 0 for row in rows)):
            raise ValueError("Point-mass law must use analytic scores with zero draws")
    elif protocol["draw_semantics"] == "mean_only":
        if (protocol.get("prediction_kind") != "mean_only" or protocol.get("residual_likelihood_status") != "unbound"
                or protocol.get("missing_metric_reason") != "residual_likelihood_unbound"
                or protocol["crps_estimator"] != "unavailable_residual_likelihood_unbound"
                or protocol["quantile_interpolation"] != "unavailable_residual_likelihood_unbound"
                or any(row["draw_count"] != 0 or row["status"] == "defined" for row in rows)
                or any(row["metrics"][name] is not None for row in rows
                       for name in ("crps","predictive_coverage","predictive_calibration_error"))):
            raise ValueError("Mean-only unbound residual requires zero predictive draws and unavailable probabilistic metrics")
    elif (protocol["crps_estimator"] != "off_diagonal_finite_ensemble"
          or any(row["status"] == "defined" and row["draw_count"] < 2 for row in rows)):
        raise ValueError("Native Monte Carlo estimator requires at least two genuine draws")
    if any(not set(schema["episode_required_fields"]) <= set(row) for row in rows):
        raise ValueError("Missing native episode score/identity fields")
    if (counts["episode_count"] != len(rows)
            or counts["defined_episode_count"] != sum(row["status"] == "defined" for row in rows)
            or counts["failed_episode_count"] != sum(row["status"] == "failed" for row in rows)
            or counts["undefined_episode_count"] != sum(row["status"] in UNDEFINED_STATUSES for row in rows)
            or counts["group_count"] != len({row["family_uid"] for row in rows})
            or counts.get("partially_defined_episode_count", 0) != sum(row["status"] == "residual_unbound" for row in rows)
            or payload["status"] != ("complete" if all(row["status"] == "defined" for row in rows) and protocol["draw_semantics"] != "mean_only" else "incomplete")):
        raise ValueError("Native metric status/episode/group denominator mismatch")
    for name in METRIC_NAMES:
        source_name = "predictive_coverage" if name == "predictive_calibration_error" else name
        count = sum(row["metrics"][source_name] is not None for row in rows)
        if payload["metrics"][name]["count"] != count or (payload["metrics"][name]["mean"] is None) != (count == 0):
            raise ValueError("Native endpoint denominator mismatch")
    if payload["metrics"]["predictive_coverage"]["count"]:
        expected = abs(payload["metrics"]["predictive_coverage"]["mean"] - payload["protocol"]["nominal_coverage"])
        if payload["metrics"]["predictive_calibration_error"]["mean"] != expected:
            raise ValueError("Native calibration error must follow aggregate coverage")
    json.dumps(payload, allow_nan=False)
