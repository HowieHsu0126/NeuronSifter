"""Actual native scale escalation with frozen checkpoints and target-free stopping."""
from __future__ import annotations

import copy
import hashlib
import json
import socket
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from Libs.utils.model.native_training_selection import training_checkpoint_path
from Libs.utils.data.native_diffusion_data import load_family_registries
from Libs.utils.data.raw_data_registry import load_config, resolve_repo_path, write_json
from Libs.utils.evaluation.native_intervention_decisions import (
    NativePublicCNSLoss, compiled_action_burden, summarize_posterior_action_loss, validate_public_loss_reference,
)
from Libs.utils.evaluation.native_intervention_evaluation import _ranking, loss_intervals, training_record_identities
from Libs.utils.evaluation.native_comparator_evaluation import load_native_comparator, declared_dataset
from Libs.utils.exp.experiment_manifests import resolve_training_seed
from Libs.utils.exp.native_comparator_training import load_frozen_native_belief, prepare_comparator_group
from Libs.utils.model.native_comparator_inputs import load_native_prediction_inputs
from Libs.utils.model.native_scientific_source import selected_group_definitions, training_data_scope
from Libs.utils.model.native_direct_mean import file_sha256


def scale_routing_policy(config: dict) -> dict:
    policy = config["native_scale_routing"]
    threshold = policy["fixed_threshold"]
    if (policy["schema_version"] != "native_scale_routing.v1"
            or policy["methods"] not in (["fixed_threshold", "fixed_route"], ["fixed_threshold", "fixed_route", "adaptive_scale"])
            or policy["stage_order"] != ["student", "large", "xl"]
            or policy["fixed_route"] != policy["stage_order"]
            or policy["threshold_statistic"] != "selected_action_optimal_posterior_mass"
            or policy["tie_break"] != "declared_action_order"
            or type(threshold) not in (int, float) or not math.isfinite(threshold)
            or not 0 < threshold <= 1
            or config["native_scientific_training"].get("source_scope", "all") != "all"):
        raise ValueError("Native scale routing requires its fixed full-mainline policy")
    return policy


def execute_scale_route(method: str, policy: dict, weights: torch.Tensor,
                        forecast: Callable[[str, dict], dict], trace: list[dict], *,
                        thresholds: dict | None = None) -> dict:
    """Only request the next forecast if this rule actually escalates.

    The callback receives no held outcome. Its loss draws retain the same
    complete tuple weights and the same declared supported action ordering.
    """
    if method not in policy["methods"]:
        raise ValueError("Unknown native scale route")
    if method == "adaptive_scale":
        if (not isinstance(thresholds, dict) or set(thresholds) != {"student", "large"}
                or any(type(value) not in (int, float) or not math.isfinite(value)
                       or not (0 <= value <= 1 or value == 2.) for value in thresholds.values())):
            raise ValueError("Adaptive routing requires its two frozen finite development thresholds")
        thresholds = dict(thresholds)
    count = None
    for stage in policy["stage_order"]:
        event = {"stage": stage, "status": "started"}
        trace.append(event)
        started = time.perf_counter()
        try:
            prediction = forecast(stage, event)
            decision = summarize_posterior_action_loss(prediction["loss"], weights)
            actions = prediction["loss"].shape[-1]
            if count is not None and actions != count:
                raise ValueError("Scale escalation changed the supported candidate set")
            count = actions
            event.update(status="defined",
                predictive_risks=decision["predictive_risks"].cpu().tolist(),
                selected_supported_index=decision["selected_index"],
                selected_optimal_mass=decision["selected_optimal_mass"],
                tied_optimal_mass=decision["tied_optimal_mass"])
        except (ValueError, KeyError, RuntimeError, FloatingPointError, OSError) as error:
            event.update(status="failed", failure=type(error).__name__+": "+str(error))
            raise
        finally:
            event["stage_elapsed_seconds"] = time.perf_counter()-started
            event["cold_checkpoint_load_seconds"] = (event.get("model_load_seconds", 0.)
                if event.get("checkpoint_loaded", False) else 0.)
            event["resident_query_seconds"] = event["stage_elapsed_seconds"]-event["cold_checkpoint_load_seconds"]
        stop = (method == "fixed_threshold"
                and decision["selected_optimal_mass"] >= policy["fixed_threshold"])
        if method == "adaptive_scale" and stage in thresholds:
            event["stop_threshold"] = thresholds[stage]
            stop = decision["selected_optimal_mass"] >= thresholds[stage]
        if stop or stage == policy["stage_order"][-1]:
            reason = ("development_fitted_mass_threshold" if method == "adaptive_scale"
                      else "fixed_posterior_mass_threshold") if stop else "declared_route_exhausted"
            return {"stage": stage, "prediction": prediction, "decision": decision, "stop_reason": reason}
    raise RuntimeError("Empty native scale route")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _stage_configurations(config: dict) -> dict:
    policy = scale_routing_policy(config)
    if set(policy["checkpoints"]) != set(policy["stage_order"]):
        raise ValueError("Every scale requires its explicit selected config and training manifest")
    seed_binding = resolve_training_seed(config, require_selected=True)
    result = {}
    for stage in policy["stage_order"]:
        spec = policy["checkpoints"][stage]
        selected = load_config(str(resolve_repo_path(spec["config_path"])))
        identity = selected.get("native_profile_identity", {})
        expected = config["manuscript_native_protocol"]["native_method_profiles"][stage]
        if (identity.get("method") != stage or identity.get("percentage") != 100
                or identity.get("checkpoint_scale") != expected["checkpoint_scale"]
                or resolve_training_seed(selected, require_selected=True) != seed_binding
                or selected["native_scientific_training"].get("source_scope", "all") != "all"
                or selected["native_scientific_training"]["predictive_budget"]
                != config["native_scientific_training"]["predictive_budget"]):
            raise ValueError("Scale checkpoint configuration differs from the common seed/domain/budget")
        manifest = resolve_repo_path(spec["training_manifest"])
        if not manifest.is_file():
            raise FileNotFoundError("Missing selected "+stage+" training manifest: "+str(manifest))
        method = selected["manuscript_native_protocol"].get("checkpoint_reuse", {}).get(stage, stage)
        result[stage] = (selected, method, manifest)
    return result


@torch.no_grad()
def evaluate_native_scale_routing(config: dict, method: str, conditioning_config: Path,
                                  conditioning_manifest: Path, dataset, declarations: dict,
                                  reference: dict, output: Path, device: torch.device, *,
                                  adaptive_policy: dict | None = None,
                                  selected_calibration: dict | None = None,
                                  calibration_reservation: str | None = None,
                                  frozen_binding: dict | None = None) -> dict:
    """Run the requested route, then score its frozen selection on held targets."""
    started = time.perf_counter()
    policy = scale_routing_policy(config)
    if method not in policy["methods"]:
        raise ValueError("Unknown native scale comparator")
    if calibration_reservation is not None and (method != "adaptive_scale"
            or calibration_reservation not in {"fit", "check"}):
        raise ValueError("Scale calibration cannot use threshold reservations or unfrozen controls")
    if method == "adaptive_scale":
        if adaptive_policy is None or frozen_binding is None:
            raise ValueError("Adaptive SCALE needs its frozen policy and current bindings")
        validate_scale_artifact(adaptive_policy, "policy_id", frozen_binding)
    Path(output).mkdir(parents=True, exist_ok=True)
    path = Path(output)/"native_scale_evaluation.json"
    if path.exists():
        raise FileExistsError("Native scale evaluation attempt already exists")
    settings = _stage_configurations(config)
    loss_function = NativePublicCNSLoss(reference, config)
    belief, conditioning = load_frozen_native_belief(
        config, conditioning_config, conditioning_manifest, device)
    registries = load_family_registries(config)
    models, bindings, rows = {}, {}, []
    report = {"schema_version": "native_scale_evaluation.v1", "method": method,
        "routing_policy": copy.deepcopy(policy), "conditioning": conditioning,
        "reference_id": reference["reference_id"], "upstream_checkpoints": bindings,
        "declared_group_count": len(declarations), "groups": rows, "status": "started",
        "selection_rule": "minimum_expected_shared_public_CNS_loss",
        "truth_boundary": "initial_support_and_programs_then_route_then_held_target_scoring",
        "timing_scope": "actual_execution_including_lazy_model_loads_and_target_scoring",
        "posterior_mass_is_calibrated_acceptance": False,
        "independent_route_calibration_qualified": False,
        "scientific_twin": "NOT_RUN", "formal_qualification": False}
    report["frozen_scale_binding"] = frozen_binding
    report["adaptive_policy_id"] = adaptive_policy["policy_id"] if adaptive_policy else None
    report["selected_loss_calibration_id"] = selected_calibration.get(
        "calibration_id", selected_calibration.get("fit_calibration_id")) if selected_calibration else None
    report["cost_basis"] = "shared_inputs_plus_resident_reached_stages_excluding_cold_loads"
    report["cold_costs"] = "initialization_and_checkpoint_loads_reported_separately"
    report["large_only_cost_reference"] = "same_resident_Large_call_on_same_inputs_hardware_and_sample_budget"
    report["independent_selected_loss_calibration_complete"] = bool(
        selected_calibration and selected_calibration.get("independent_check_complete"))
    report["initialization_seconds"] = time.perf_counter()-started
    seed = int(resolve_training_seed(config, require_selected=True)["model_seed"])
    budget = config["native_scientific_training"]["predictive_budget"]
    for index, uid in enumerate(dataset.group_uids):
        row = {"group_uid": uid, "status": "started", "route": []}
        rows.append(row)
        group_started = time.perf_counter()
        try:
            declaration = declarations[uid]
            expected_role = "calibration" if calibration_reservation else declaration["role"]
            if (expected_role not in {"development", "validation", "calibration"}
                    or declaration["role"] != expected_role
                    or declaration.get("calibration_reservation") != calibration_reservation):
                raise ValueError("Scale comparison role/reservation differs from its frozen purpose")
            row.update(role=declaration["role"], calibration_reservation=calibration_reservation)
            inputs = load_native_prediction_inputs(config, declaration, belief, registries)
            weights = inputs.weights[0]
            burdens = []
            for action, regimen in enumerate(inputs.regimen_uids):
                values = [compiled_action_burden(records[action], inputs.grid["family_uid"], config)
                          for records in inputs.programs["records"][0]]
                if any(value != values[0] for value in values[1:]):
                    raise ValueError("Intervention burden differs across complete posterior tuples")
                burdens.append(dict(values[0], regimen_uid=regimen))
            feasible = [i for i, item in enumerate(burdens) if item["status"] == "defined"]
            row.update(family_uid=inputs.grid["family_uid"], partition_uid=declaration["partition_uid"],
                source_uid=inputs.posterior.support.source_uid, burdens=burdens,
                declared_action_count=len(burdens), supported_action_count=len(feasible),
                withheld_action_count=len(burdens)-len(feasible),
                posterior_weights=weights.cpu().tolist())
            if not feasible:
                row.update(status="insufficient_evidence", reason="no_supported_public_loss_action")
                continue
            cost = weights.new_tensor([burdens[i]["normalized_burden"] for i in feasible])
            source_mask = inputs.validity.reshape(*inputs.group_shape, *inputs.validity.shape[1:])[0, 0]

            _synchronize(device)
            row["shared_input_seconds"] = time.perf_counter()-group_started

            def forecast(stage: str, event: dict) -> dict:
                _synchronize(device)
                load_started = time.perf_counter()
                event["checkpoint_loaded"] = stage not in models
                if event["checkpoint_loaded"]:
                    selected, actual_method, manifest = settings[stage]
                    model, reused, binding = load_native_comparator(
                        selected, actual_method, conditioning_config, conditioning_manifest,
                        manifest, dataset, device, frozen_conditioning=(belief, conditioning))
                    if reused is not belief:
                        raise ValueError("Scale routing replaced the common frozen posterior")
                    models[stage], bindings[stage] = model, binding
                    bindings[stage]["actual_parameter_count"] = sum(p.numel() for p in model.parameters())
                _synchronize(device)
                event["model_load_seconds"] = time.perf_counter()-load_started
                event["sampling_calls"] = 1
                sample_started = time.perf_counter()
                law = models[stage].sample_native(inputs, belief.field_normalization,
                    budget["conditional_draws_per_particle"], budget["categorical_reverse_steps"],
                    seed+config["native_scientific_training"]["validation_stream_offset"]+index)
                _synchronize(device)
                event["sampling_seconds"] = time.perf_counter()-sample_started
                samples = law["particle_trajectory_samples"][:, 0].transpose(0, 1)
                if (samples.shape[:3] != (len(weights), budget["conditional_draws_per_particle"], len(burdens))
                        or samples.shape[3:] != source_mask.shape[1:]):
                    raise ValueError("Scale forecast lost complete tuple/draw/action/native-grid axes")
                event["particle_action_paths"] = math.prod(samples.shape[:3])
                event["conditional_draws_per_particle"] = samples.shape[1]
                prediction = loss_function(samples[:, :, feasible], source_mask[feasible], inputs.grid, cost)
                prediction["weighted_clipped_fractions"] = (
                    prediction["clipped_fraction"].double().mean(1)*weights.double()[:, None, None]).sum(0)/weights.double().sum()
                return prediction

            routed = execute_scale_route(method, policy, weights, forecast, row["route"],
                thresholds=adaptive_policy["thresholds"] if adaptive_policy else None)
            decision, prediction = routed["decision"], routed["prediction"]
            choice = decision["selected_index"]
            row.update(selected_stage=routed["stage"], stop_reason=routed["stop_reason"],
                selected_regimen_uid=inputs.regimen_uids[feasible[choice]],
                supported_regimen_uids=[inputs.regimen_uids[i] for i in feasible],
                predictive_risks=decision["predictive_risks"].cpu().tolist(),
                predictive_clipped_fractions=prediction["weighted_clipped_fractions"].cpu().tolist())
            if selected_calibration is not None:
                family_quantiles = selected_calibration["quantiles"][inputs.grid["family_uid"]]
                row["selected_loss_upper_bounds"] = {level: selected_loss_upper_bound(
                    float(decision["predictive_risks"][choice]), quantile, prediction["maximum_supported_loss"])
                    for level, quantile in family_quantiles.items()}
                row["calibration_application"] = ("in_distribution" if declaration.get("source_variant", "primary") == "primary"
                    and declaration.get("ood_class", "in_distribution") == "in_distribution" else "shifted_unretuned")
            # Open targets only after route, action and any calibrated bound are fixed.
            group = dataset[index]["native_joint_group"]
            checked, target, mask, _ = prepare_comparator_group(group, belief, config, device)
            if (checked.sample_uids != inputs.sample_uids or checked.regimen_uids != inputs.regimen_uids
                    or checked.posterior_uid != inputs.posterior_uid
                    or not torch.equal(checked.weights, inputs.weights)):
                raise ValueError("Held target group differs from the frozen routed prediction")
            observed = loss_function(target[0, feasible], mask[0, feasible], inputs.grid, cost)
            metrics = _ranking(decision["predictive_risks"].cpu().numpy(), observed["loss"].cpu().numpy())
            if metrics.pop("selected_supported_index") != choice:
                raise ValueError("Held outcomes changed the fixed scale-route selection")
            metrics["supported_set_intervention_regret"] = metrics.pop("intervention_regret")
            metrics["optimal_action_recall"] = float(observed["loss"][choice] == observed["loss"].min())
            row.update(status="defined" if len(feasible) == len(burdens) else "defined_on_supported_actions",
                observed_public_losses=observed["loss"].cpu().tolist(),
                observed_clipped_fractions=observed["clipped_fraction"].cpu().tolist(),
                loss_intervals_by_nominal=loss_intervals(prediction["loss"], weights, observed["loss"],
                    config["native_controlled_comparators"]["nominal_levels"]),
                metrics=metrics, maximum_supported_loss=prediction["maximum_supported_loss"])
            if "selected_loss_upper_bounds" in row:
                row["selected_loss_bound_coverage"] = {level: float(observed["loss"][choice]) <= upper
                    for level, upper in row["selected_loss_upper_bounds"].items()}
        except (ValueError, KeyError, RuntimeError, FloatingPointError, OSError) as error:
            row.update(status="failed", failure={"type": type(error).__name__, "message": str(error)})
        finally:
            _synchronize(device)
            row["elapsed_seconds"] = time.perf_counter()-group_started
            row["sampling_calls"] = sum(stage.get("sampling_calls", 0) for stage in row["route"])
            row["particle_action_paths"] = sum(stage.get("particle_action_paths", 0) for stage in row["route"])
            row["resident_inference_seconds"] = row.get("shared_input_seconds", 0.)+sum(
                stage.get("resident_query_seconds", 0.) for stage in row["route"])
            row["cold_checkpoint_load_seconds"] = sum(stage.get("cold_checkpoint_load_seconds", 0.) for stage in row["route"])
            write_json(path, report)
    seen = [row["group_uid"] for row in rows]
    if len(seen) != len(set(seen)) or set(seen) != set(declarations):
        report["denominator_failure"] = "missing_duplicate_or_undeclared_groups"
    report["defined_group_count"] = sum(row["status"] in {"defined", "defined_on_supported_actions"} for row in rows)
    report["fully_supported_group_count"] = sum(row["status"] == "defined" for row in rows)
    report["sampling_calls"] = sum(row["sampling_calls"] for row in rows)
    report["particle_action_paths"] = sum(row["particle_action_paths"] for row in rows)
    report["escalated_group_count"] = sum(len(row["route"]) > 1 for row in rows)
    report["escalation_rate"] = report["escalated_group_count"]/len(declarations) if declarations else None
    report["resident_inference_seconds"] = sum(row["resident_inference_seconds"] for row in rows)
    report["cold_checkpoint_load_seconds"] = sum(row["cold_checkpoint_load_seconds"] for row in rows)
    report["elapsed_seconds"] = time.perf_counter()-started
    report["status"] = ("complete" if declarations and report["fully_supported_group_count"] == len(declarations)
                        and not report.get("denominator_failure") else "incomplete")
    write_json(path, report)
    return report


ADAPTIVE_METHOD = "adaptive_scale"
ALWAYS_STOP_THRESHOLD = 0.0
ALWAYS_ESCALATE_THRESHOLD = 2.0
DEVELOPMENT_GROUPS_PER_FAMILY = 200
CALIBRATION_GROUPS_PER_FAMILY = 512


def adaptive_scale_settings(config: dict) -> dict:
    """The newly declared operational rule; not a recovered manuscript algorithm."""
    settings = scale_routing_policy(config).get("adaptive")
    if (not isinstance(settings, dict)
            or settings["definition_status"] != "prospective_operational_definition"
            or settings["fit_role"] != "development"
            or settings["development_groups_per_family"] != DEVELOPMENT_GROUPS_PER_FAMILY
            or settings["calibration_groups_per_family"] != CALIBRATION_GROUPS_PER_FAMILY
            or settings["cost_scope"] != "shared_inputs_plus_resident_reached_stages_excluding_cold_loads"
            or settings["objective"] != "minimum_mean_controlled_oracle_regret_under_large_only_mean_cost"
            or settings["nominal_levels"] != [0.8, 0.85, 0.9, 0.95]):
        raise ValueError("Adaptive SCALE requires its frozen prospective objective and full group counts")
    return settings


def optimize_scale_thresholds(masses: np.ndarray, regrets: np.ndarray,
                              stage_costs: np.ndarray, shared_costs: np.ndarray) -> dict:
    """Exact finite search over two global thresholds; artifact fitting checks roles below.

    All stage outputs/costs are measured on the same development groups. Prefix
    sums evaluate all Large thresholds for each Student threshold, retaining
    every group without an O(number_of_pairs * number_of_groups) rescan.
    """
    masses, regrets, stage_costs, shared_costs = (
        np.asarray(value, dtype=np.float64) for value in (masses, regrets, stage_costs, shared_costs))
    n = len(masses)
    if (not n or masses.shape != (n, 2) or regrets.shape != (n, 3)
            or stage_costs.shape != (n, 3) or shared_costs.shape != (n,)
            or not all(np.isfinite(value).all() for value in (masses, regrets, stage_costs, shared_costs))
            or (masses < 0).any() or (masses > 1).any() or (regrets < 0).any()
            or (stage_costs <= 0).any() or (shared_costs < 0).any()):
        raise ValueError("Adaptive fitting requires complete finite development masses, regrets and measured costs")
    candidates = [np.unique(np.concatenate(([ALWAYS_STOP_THRESHOLD], masses[:, i],
                                            [ALWAYS_ESCALATE_THRESHOLD]))) for i in range(2)]
    order = np.argsort(masses[:, 1], kind="stable")
    positions = np.searchsorted(masses[order, 1], candidates[1], side="left")
    large_only = float(np.mean(shared_costs+stage_costs[:, 1]))
    common = float(np.sum(shared_costs+stage_costs[:, 0]))
    best = None
    for student_threshold in candidates[0]:
        reached = masses[:, 0] < student_threshold
        reached_sorted = reached[order]
        extra_cost = np.concatenate(([0.], np.cumsum(reached_sorted*stage_costs[order, 2])))
        extra_regret = np.concatenate(([0.], np.cumsum(reached_sorted*regrets[order, 2])))
        # Nonnegative prefix/suffix sums avoid cancellation between large regrets.
        large_regret = np.concatenate((np.cumsum((reached_sorted*regrets[order, 1])[::-1])[::-1], [0.]))
        costs = (common+stage_costs[reached, 1].sum()+extra_cost[positions])/n
        risks = (regrets[~reached, 0].sum()+extra_regret[positions]+large_regret[positions])/n
        eligible = np.flatnonzero(costs <= large_only)
        if not len(eligible):
            continue
        local = eligible[np.lexsort((candidates[1][eligible], costs[eligible], risks[eligible]))[0]]
        candidate = (float(risks[local]), float(costs[local]), float(student_threshold),
                     float(candidates[1][local]))
        if best is None or candidate < best:
            best = candidate
    result = {"candidate_policy_count": int(len(candidates[0])*len(candidates[1])),
              "development_group_count": n, "large_only_mean_resident_cost_seconds": large_only,
              "endpoint_thresholds": {"always_stop": ALWAYS_STOP_THRESHOLD,
                                      "always_escalate": ALWAYS_ESCALATE_THRESHOLD},
              "tie_break": ["lower_mean_regret", "lower_mean_cost", "student_threshold", "large_threshold"]}
    if best is None:
        return dict(result, status="infeasible", reason="no_policy_meets_unchanged_large_only_mean_cost")
    student_stops = masses[:, 0] >= best[2]
    large_stops = ~student_stops & (masses[:, 1] >= best[3])
    counts = {"student": int(student_stops.sum()), "large": int(large_stops.sum()),
              "xl": int((~student_stops & ~large_stops).sum())}
    return dict(result, status="complete", thresholds={"student": best[2], "large": best[3]},
                mean_selected_controlled_oracle_regret=best[0],
                mean_resident_cost_seconds=best[1], selected_stage_counts=counts)


def _scale_digest(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def seal_scale_artifact(payload: dict, key: str) -> dict:
    if key in payload:
        raise ValueError("Scale artifact already carries its immutable identity")
    return dict(payload, **{key: _scale_digest(payload)})


def validate_scale_artifact(payload: dict, key: str, binding: dict, *,
                            insufficient_schema: dict | None = None) -> None:
    # Writers may validate the separately declared insufficient shape. Ordinary
    # policy/calibrator consumers never opt in and still require a sealed result.
    if insufficient_schema is not None and payload.get("status") == "insufficient_data":
        rule = insufficient_schema["status_variants"]["insufficient_data"]
        if (payload.get("schema_version") != insufficient_schema["schema_version"]
                or any(name not in payload for name in rule["required_fields"])
                or any(type(payload.get(name)) is not int or payload[name] != 0 for name in rule["zero_fields"])
                or any(payload.get(name) is not False for name in rule["false_fields"])
                or any(name in payload for name in rule["forbidden_fields"])
                or payload.get("scientific_twin") != "NOT_RUN"
                or payload["denominator"].get("status") != "insufficient_data"
                or not payload["denominator"].get("roles")):
            raise ValueError("Invalid insufficient SCALE shape, count evidence or zero-call boundary")
        return
    schemas = {"policy_id": "native_adaptive_scale_policy.v1", "calibration_id": "native_scale_selected_calibration.v1"}
    if (payload.get("schema_version") != schemas.get(key)
            or payload.get("status") != "complete" or payload.get("binding") != binding
            or payload.get(key) != _scale_digest({name: value for name, value in payload.items() if name != key})):
        raise ValueError("Scale artifact is incomplete or its frozen identity/binding changed")


def scale_role_contract(config: dict) -> tuple[dict, dict]:
    """Check full prospective denominators before datasets, models or labels."""
    settings = adaptive_scale_settings(config)
    families = tuple(config["native_observable_bridge"]["quantity_profile"]["families"])
    if len(families) != 4:
        raise ValueError("Adaptive SCALE requires all four native domain families")
    selections, reports = {}, {}
    seen = set()
    for name, role, reservation, required in (
        ("development", "development", None, settings["development_groups_per_family"]),
        ("fit", "calibration", "fit", settings["calibration_groups_per_family"]),
        ("check", "calibration", "check", settings["calibration_groups_per_family"]),
        ("threshold", "calibration", "threshold", settings["calibration_groups_per_family"]),
    ):
        try:
            rows = selected_group_definitions(config, role, reservation)
        except ValueError as error:
            rows = []
            selection_error = str(error)
        else:
            selection_error = None
        ids = [row["sample_uid"] for row in rows]
        counts = {family: sum(row["family_uid"] == family for row in rows) for family in families}
        valid = (len(ids) == len(set(ids)) and not seen.intersection(ids)
                 and all(count == required for count in counts.values())
                 and len(rows) == required*len(families)
                 and all(row["role"] == role and row["calibration_reservation"] == reservation
                         and row.get("source_variant", "primary") == "primary"
                         and row.get("ood_class", "in_distribution") == "in_distribution" for row in rows))
        reports[name] = {"status": "complete" if valid else "insufficient_or_invalid_groups",
                         "required_groups_per_family": required, "required_total": required*len(families),
                         "declared_total": len(rows), "declared_by_family": counts}
        if selection_error:
            reports[name]["selection_error"] = selection_error
        selections[name] = {row["sample_uid"]: row for row in rows}
        seen.update(ids)
    return ({"status": "complete" if all(row["status"] == "complete" for row in reports.values())
             else "insufficient_data", "roles": reports,
             "threshold_role": "reserved_not_used_for_routing_or_selected_bound_fit_check"}, selections)


def selected_loss_quantile(errors: list[float], nominal_level: float) -> dict:
    """Manuscript ceil((n+1)*nominal) order statistic, including its infinite case."""
    values = np.asarray(errors, dtype=np.float64)
    if (values.ndim != 1 or not np.isfinite(values).all()
            or not math.isfinite(nominal_level) or not 0 < nominal_level < 1):
        raise ValueError("Selected-loss calibration needs finite errors and a valid nominal level")
    rank = math.ceil((len(values)+1)*nominal_level)
    result = {"nominal_level": nominal_level, "sample_count": len(values), "order_statistic_rank": rank}
    if rank > len(values):
        return dict(result, kind="positive_infinity")
    return dict(result, kind="finite", value=float(np.partition(values, rank-1)[rank-1]))


def selected_loss_upper_bound(predicted_risk: float, quantile: dict, maximum_loss: float) -> float:
    if not math.isfinite(predicted_risk) or not math.isfinite(maximum_loss) or maximum_loss <= 0:
        raise ValueError("Selected-loss bound requires finite prediction and a positive declared loss bound")
    if quantile["kind"] == "positive_infinity":
        return maximum_loss
    if quantile["kind"] != "finite" or not math.isfinite(quantile["value"]):
        raise ValueError("Unknown selected-loss quantile representation")
    return min(maximum_loss, max(0., predicted_risk+quantile["value"]))


def _complete_scale_rows(report: dict, declarations: dict, role: str, reservation) -> list[dict]:
    rows = report.get("groups", [])
    ids = [row.get("group_uid") for row in rows]
    if (report.get("status") != "complete" or len(ids) != len(set(ids))
            or set(ids) != set(declarations) or not ids):
        raise ValueError("Scale collection lost failed, unsupported, duplicate or missing declared groups")
    counts = {}
    for declared in declarations.values():
        counts[declared["family_uid"]] = counts.get(declared["family_uid"], 0)+1
    expected = DEVELOPMENT_GROUPS_PER_FAMILY if role == "development" else CALIBRATION_GROUPS_PER_FAMILY
    if len(counts) != 4 or any(count != expected for count in counts.values()):
        raise ValueError("Scale role is below the unchanged per-family production denominator")
    for row in rows:
        declared = declarations[row["group_uid"]]
        if (row["status"] != "defined" or row.get("role") != role
                or row.get("calibration_reservation") != reservation
                or declared["role"] != role or declared["calibration_reservation"] != reservation
                or row["family_uid"] != declared["family_uid"]):
            raise ValueError("Scale collection role/reservation/domain differs from its declaration")
    return rows


def fit_scale_policy_records(report: dict, declarations: dict, binding: dict,
                             denominator: dict) -> dict:
    """Fit from complete development execution only; calibration labels are absent."""
    if denominator["status"] != "complete":
        raise ValueError("Adaptive fitting requires the full independent group denominators")
    rows = _complete_scale_rows(report, declarations, "development", None)
    if report.get("frozen_scale_binding") != binding:
        raise ValueError("Development collection source/checkpoint/config bindings changed")
    if report["method"] != "fixed_route":
        raise ValueError("Development collection must measure every declared stage once")
    masses, regrets, costs, shared = [], [], [], []
    for row in rows:
        stages = row["route"]
        if [stage["stage"] for stage in stages] != ["student", "large", "xl"]:
            raise ValueError("Development collection is missing a reached-stage cost/decision")
        observed = np.asarray(row["observed_public_losses"], dtype=np.float64)
        if not np.isfinite(observed).all() or not len(observed):
            raise ValueError("Nonfinite or empty development controlled outcomes")
        selections = [stage["selected_supported_index"] for stage in stages]
        if any(type(index) is not int or not 0 <= index < len(observed) for index in selections):
            raise ValueError("Development selected action is outside the unchanged candidate set")
        masses.append([stage["selected_optimal_mass"] for stage in stages[:2]])
        regrets.append([float(observed[index]-observed.min()) for index in selections])
        costs.append([stage["resident_query_seconds"] for stage in stages])
        shared.append(row["shared_input_seconds"])
    result = optimize_scale_thresholds(np.array(masses), np.array(regrets), np.array(costs), np.array(shared))
    artifact = dict(result, schema_version="native_adaptive_scale_policy.v1",
        definition_status="prospective_operational_definition", binding=binding,
        fit_role="development", fit_group_uids=[row["group_uid"] for row in rows],
        denominator=denominator, development_collection=report,
        calibration_used_for_routing=False, additional_optimizer_updates=0,
        scientific_twin="NOT_RUN", formal_qualification=False)
    return seal_scale_artifact(artifact, "policy_id") if artifact["status"] == "complete" else artifact


def fit_selected_scale_calibration(report: dict, declarations: dict, policy: dict,
                                   binding: dict, nominal_levels: list[float]) -> dict:
    validate_scale_artifact(policy, "policy_id", binding)
    rows = _complete_scale_rows(report, declarations, "calibration", "fit")
    if (report.get("frozen_scale_binding") != binding or report["method"] != ADAPTIVE_METHOD
            or report.get("adaptive_policy_id") != policy["policy_id"]
            or set(declarations).intersection(policy["fit_group_uids"])):
        raise ValueError("Selected-loss calibration does not follow the frozen development policy")
    errors = {}
    for row in rows:
        choice = row["supported_regimen_uids"].index(row["selected_regimen_uid"])
        error = row["observed_public_losses"][choice]-row["predictive_risks"][choice]
        errors.setdefault(row["family_uid"], []).append(error)
    if any(len(values) != CALIBRATION_GROUPS_PER_FAMILY for values in errors.values()) or len(errors) != 4:
        raise ValueError("Selected-loss calibration requires512 independent groups in every family")
    artifact = {"schema_version": "native_scale_selected_calibration.v1", "status": "fit_complete",
        "binding": binding, "policy_id": policy["policy_id"], "fit_group_uids": list(declarations),
        "fit_role": "calibration_fit", "grouping": "independent_groups_within_native_family",
        "quantiles": {family: {str(level): selected_loss_quantile(values, level)
                              for level in nominal_levels} for family, values in errors.items()},
        "routing_changed_by_calibration": False, "threshold_role_used": False,
        "independent_check_complete": False, "scientific_twin": "NOT_RUN", "formal_qualification": False}
    return seal_scale_artifact(artifact, "fit_calibration_id")


def check_selected_scale_calibration(report: dict, declarations: dict, fitted: dict,
                                     policy: dict, binding: dict) -> dict:
    validate_scale_artifact(policy, "policy_id", binding)
    if (fitted.get("status") != "fit_complete" or fitted.get("binding") != binding
            or fitted.get("policy_id") != policy["policy_id"]
            or fitted.get("fit_calibration_id") != _scale_digest(
                {key: value for key, value in fitted.items() if key != "fit_calibration_id"})):
        raise ValueError("Selected calibration fit identity changed before its independent check")
    rows = _complete_scale_rows(report, declarations, "calibration", "check")
    if (report.get("frozen_scale_binding") != binding or report.get("adaptive_policy_id") != policy["policy_id"]
            or set(declarations).intersection(fitted["fit_group_uids"])
            or set(declarations).intersection(policy["fit_group_uids"])):
        raise ValueError("Calibration check overlaps fitting or uses a different frozen route")
    checks = {}
    for row in rows:
        family = row["family_uid"]
        choice = row["supported_regimen_uids"].index(row["selected_regimen_uid"])
        actual, predicted = row["observed_public_losses"][choice], row["predictive_risks"][choice]
        if not math.isfinite(actual) or not math.isfinite(predicted):
            raise ValueError("Nonfinite independent calibration outcome/prediction")
        family_checks = checks.setdefault(family, {level: [] for level in fitted["quantiles"][family]})
        for level, quantile in fitted["quantiles"][family].items():
            upper = selected_loss_upper_bound(predicted, quantile, row["maximum_supported_loss"])
            family_checks[level].append(float(actual <= upper))
    if len(checks) != 4 or any(len(values) != CALIBRATION_GROUPS_PER_FAMILY
                              for family in checks.values() for values in family.values()):
        raise ValueError("Independent calibration check requires512 groups in every family")
    coverage = {family: {level: {"count": len(values), "covered": int(sum(values)),
                                "coverage": float(np.mean(values)),
                                "nominal_level": float(level)}
                        for level, values in levels.items()} for family, levels in checks.items()}
    artifact = dict(fitted, status="complete", check_role="calibration_check",
        check_group_uids=list(declarations), independent_check_complete=True, check_coverage=coverage,
        coverage_is="empirical_check_not_an_added_pass_threshold_or_shift_guarantee")
    return seal_scale_artifact(artifact, "calibration_id")


def _selected_checkpoint_identity(config_path: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    selection = manifest.get("selection", {})
    if selection.get("selection_role") != "development" or not selection.get("best"):
        raise ValueError("Adaptive SCALE requires development-selected upstream checkpoints")
    checkpoint = training_checkpoint_path(manifest_path,selection,selection["best"]["checkpoint_path"])
    return {"config_path": str(config_path), "config_sha256": file_sha256(config_path),
            "manifest_path": str(manifest_path), "manifest_sha256": file_sha256(manifest_path),
            "checkpoint_path": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
            "sufficient_training": selection.get("sufficient_training") is True}


def frozen_scale_binding(config: dict, conditioning_config: Path, conditioning_manifest: Path,
                         reference: dict, device: torch.device) -> dict:
    """Bind every upstream before calibration labels, including unreached stages."""
    stages = _stage_configurations(config)
    budget = config["native_scientific_training"]["predictive_budget"]
    if (budget["complete_particles"] != 32 or budget["conditional_draws_per_particle"] != 2
            or budget["categorical_reverse_steps"] != 100 or budget["weighted_forecast_paths_per_action"] != 64):
        raise ValueError("Adaptive SCALE retains32 whole tuples,64 paths/action and100 denoising steps")
    checkpoints = {stage: _selected_checkpoint_identity(
        resolve_repo_path(config["native_scale_routing"]["checkpoints"][stage]["config_path"]), entry[2])
        for stage, entry in stages.items()}
    return {"configuration_sha256": _scale_digest(config),
            "routing_code_sha256": file_sha256(Path(__file__)),
            "source": training_data_scope(config), "reference_id": reference["reference_id"],
            "seed": resolve_training_seed(config, require_selected=True), "sample_budget": copy.deepcopy(budget),
            "conditioning": _selected_checkpoint_identity(conditioning_config, conditioning_manifest),
            "checkpoints": checkpoints,
            "hardware": {"hostname": socket.gethostname(), "device": str(device),
                         "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                         "torch_threads": torch.get_num_threads(),
                         "torch_interop_threads": torch.get_num_interop_threads()},
            "cost_scope": adaptive_scale_settings(config)["cost_scope"]}


def run_adaptive_scale_command(config: dict, command: str, conditioning_config: Path,
                               conditioning_manifest: Path, reference_path: Path, output: Path,
                               device: torch.device, *, policy_path: Path | None = None,
                               calibration_path: Path | None = None,
                               evaluation_role: str = "validation") -> dict:
    """Ordinary native fit/calibrate/evaluate path; no optimizer or source calls."""
    names = {"fit_scale_policy": "native_adaptive_scale_policy.json",
             "calibrate_scale_policy": "native_scale_selected_calibration.json",
             "evaluate_baseline": "native_scale_evaluation.json"}
    if command not in names:
        raise ValueError("Adaptive SCALE uses fit_scale_policy, calibrate_scale_policy or evaluate_baseline")
    output.mkdir(parents=True, exist_ok=True)
    destination = output/names[command]
    if destination.exists():
        raise FileExistsError("Adaptive SCALE attempt already exists")
    schemas = {"fit_scale_policy": "native_adaptive_scale_policy.v1",
               "calibrate_scale_policy": "native_scale_selected_calibration.v1",
               "evaluate_baseline": "native_scale_evaluation.v1"}
    initial = {"schema_version": schemas[command], "method": ADAPTIVE_METHOD, "command": command,
               "status": "started", "additional_optimizer_updates": 0, "source_provider_calls": 0,
               "scientific_twin": "NOT_RUN", "formal_qualification": False}
    # Counts and disjoint declarations precede checkpoint/data/outcome access.
    denominator, selections = scale_role_contract(config)
    initial["denominator"] = denominator
    if denominator["status"] != "complete":
        result = dict(initial, status="insufficient_data", reason="full_role_denominators_not_available",
                      model_calls=0, calibration_outcome_access=False)
        validate_scale_artifact(result, "", {}, insufficient_schema=config["schemas"][names[command].removesuffix(".json")])
        write_json(destination, result)
        return result
    write_json(destination, initial)
    try:
        if reference_path is None:
            raise ValueError("Adaptive SCALE requires the frozen training public-loss reference")
        reference = json.loads(reference_path.read_text())
        training, training_ids = declared_dataset(config, "train")
        validate_public_loss_reference(reference, config, set(training_ids), training_record_identities(training))
        binding = frozen_scale_binding(config, conditioning_config, conditioning_manifest, reference, device)
        if command == "fit_scale_policy":
            dataset, declarations = declared_dataset(config, "development")
            collected = evaluate_native_scale_routing(
                config, "fixed_route", conditioning_config, conditioning_manifest,
                dataset, declarations, reference, output/"development_collection", device,
                frozen_binding=binding)
            result = fit_scale_policy_records(collected, declarations, binding, denominator)
        else:
            if policy_path is None:
                raise ValueError("Adaptive SCALE requires a frozen development policy artifact")
            policy = json.loads(policy_path.read_text())
            validate_scale_artifact(policy, "policy_id", binding)
            if set(policy["fit_group_uids"]) != set(selections["development"]):
                raise ValueError("Frozen policy development identities differ from current declarations")
            if command == "calibrate_scale_policy":
                fit, fit_ids = declared_dataset(config, "calibration", reservation="fit")
                fit_report = evaluate_native_scale_routing(
                    config, ADAPTIVE_METHOD, conditioning_config, conditioning_manifest,
                    fit, fit_ids, reference, output/"fit_collection", device,
                    adaptive_policy=policy, calibration_reservation="fit", frozen_binding=binding)
                fitted = fit_selected_scale_calibration(
                    fit_report, fit_ids, policy, binding, adaptive_scale_settings(config)["nominal_levels"])
                write_json(output/"native_scale_selected_calibration_fit.json", fitted)
                check, check_ids = declared_dataset(config, "calibration", reservation="check")
                checked = evaluate_native_scale_routing(
                    config, ADAPTIVE_METHOD, conditioning_config, conditioning_manifest,
                    check, check_ids, reference, output/"check_collection", device,
                    adaptive_policy=policy, selected_calibration=fitted,
                    calibration_reservation="check", frozen_binding=binding)
                result = check_selected_scale_calibration(checked, check_ids, fitted, policy, binding)
            else:
                if calibration_path is None:
                    raise ValueError("Adaptive SCALE evaluation requires an independently checked calibration")
                calibration = json.loads(calibration_path.read_text())
                validate_scale_artifact(calibration, "calibration_id", binding)
                if (calibration["policy_id"] != policy["policy_id"]
                        or not calibration["independent_check_complete"]
                        or set(calibration["fit_group_uids"]) != set(selections["fit"])
                        or set(calibration["check_group_uids"]) != set(selections["check"])):
                    raise ValueError("Selected-loss calibration policy or fit/check identities differ")
                dataset, declarations = declared_dataset(config, evaluation_role)
                used = set(policy["fit_group_uids"]) | set(calibration["fit_group_uids"]) | set(calibration["check_group_uids"]) | set(selections["threshold"])
                if set(declarations).intersection(used):
                    raise ValueError("Adaptive evaluation overlaps policy/calibration/threshold reservations")
                result = evaluate_native_scale_routing(
                    config, ADAPTIVE_METHOD, conditioning_config, conditioning_manifest,
                    dataset, declarations, reference, output/"evaluation_collection", device,
                    adaptive_policy=policy, selected_calibration=calibration, frozen_binding=binding)
        write_json(destination, result)
        return result
    except (ValueError, KeyError, RuntimeError, FloatingPointError, OSError) as error:
        result = dict(initial, status="incomplete",
                      failure={"type": type(error).__name__, "message": str(error)})
        write_json(destination, result)
        return result
