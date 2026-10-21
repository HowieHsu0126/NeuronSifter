"""Role-preserving calibration orchestration for the frozen native direct mean."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import torch

from Libs.utils.data import native_diffusion_data as native
from Libs.utils.model.screening_model_config import native_transolver_mean_config
from Libs.utils.data.native_field_normalization import source_domain
from Libs.utils.model.native_direct_mean import (
    direct_model_identity, file_sha256, load_direct_predictor, predict_direct_mean)
from Libs.utils.model.native_transolver_residual import (
    LAW, SPECIES_COUNT, check_diagnostics, episode_statistics, fit_scales,
    information_counts, residual_draws, residual_objective)
from Libs.utils.model.joint_rollout import posterior_draw_indices
from Libs.utils.evaluation.native_rollout_metrics import (
    metric_protocol, score_native_episode, build_native_metric_payload)


def residual_policy(config: dict) -> dict:
    native_transolver_mean_config(config)  # Never extend this law to another direct predictor.
    policy = copy.deepcopy(config["paper_grade_full_suite"]["native_transolver_residual"])
    if (policy["law"] != LAW or policy["reference_time_hours"] != 672.
            or policy["source_role"] != "calibration" or policy["group_field"] != "family_uid"
            or policy["readout_kind"] != "independent_mechanistic_teacher_readout"
            or policy["missingness"] != "ignorable_given_conditions_and_mask"
            or policy["diagnostics"] != ["adjacent_retained_times", "native_graph_edges", "cross_species_pairs"]):
        raise ValueError("Unsupported frozen residual law/source/missingness/diagnostic policy")
    n = policy["numerical"]
    if (n["dtype"] != "float64" or n["eta_initialization"] != 0. or n["optimizer"] != "torch_lbfgs"
            or n["line_search"] != "strong_wolfe" or n["history_size"] != 10
            or n["gradient_tolerance"] != 1e-7 or n["change_tolerance"] != 1e-12
            or n["hessian_absolute_threshold"] != 1e-10 or n["hessian_relative_threshold"] != 1e-8
            or not 1 <= n["max_iterations"] <= 200 or not 1 <= n["max_closure_evaluations"] <= 400):
        raise ValueError("Unsupported prespecified residual numerical protocol")
    if not isinstance(policy["residual_seed"], int) or isinstance(policy["residual_seed"], bool):
        raise ValueError("Separate recorded integer residual RNG seed required")
    return policy


def _source_identity(source: dict, policy: dict) -> dict:
    declaration = source.get("residual_readout", {})
    if (declaration.get("kind") != policy["readout_kind"]
            or declaration.get("missingness") != policy["missingness"]
            or declaration.get("independent_groups") is not True or not declaration.get("evidence_ref")):
        raise ValueError("Source must declare independent mechanistic readouts, groups and ignorable masks")
    keys = ("source_uid", "public", "protected", *native.SOURCE_FIELDS, "predictive_evaluation_units", "residual_readout")
    return {key: copy.deepcopy(source.get(key)) for key in keys}


def admit_partition(config: dict, registry: dict, policy: dict) -> dict:
    # All fit/check metadata is admitted before either operation reads a target.
    from Libs.exp.evaluation.evaluate_screening_system import _native_episode_identity, _native_validity_mask

    roles = {}
    all_groups, all_samples = set(), set()
    actions = config["paper_grade_full_suite"]["native_rollout_evaluation"]["regimen_uids"]
    if not actions or len(set(actions)) != len(actions):
        raise ValueError("Explicit unique complete calibration action set required")
    if registry["grid"]["physical_time_unit"] != "hour" or len(registry["grid"]["species_id"]) != SPECIES_COUNT:
        raise ValueError("Residual calibration requires the complete 15-species hour domain")
    source_ids = []
    for operation in ("fit", "check"):
        groups = policy[operation+"_groups"]
        if not groups:
            raise FileNotFoundError("Missing predeclared complete calibration-"+operation+" groups")
        episodes, identities, metadata = [], [], []
        for group in groups:
            uid, samples = group["group_uid"], group["sample_uids"]
            if (not uid or uid in all_groups or not samples or len(set(samples)) != len(samples)
                    or any(sample in all_samples for sample in samples)):
                raise ValueError("Overlapping/empty calibration group or episode selection")
            all_groups.add(uid)
            all_samples.update(samples)
            members = [row for row in registry["teachers"] if row["family_uid"] == uid]
            if (not members or any(row["split_name"] != "calibration" or row["source_holdout_role"] != "calibration" for row in members)
                    or {(row["sample_uid"], row["regimen_uid"]) for row in members}
                    != {(sample, action) for sample in samples for action in actions}):
                raise ValueError("Incomplete group or calibration role promotion")
            records = native.select_records(registry, config, "calibration", samples, actions)
            for start in range(0, len(records), len(actions)):
                current = records[start:start+len(actions)]
                if any(row["teacher"]["qc_grade"] not in {"A", "B"} for row in current):
                    raise ValueError("Primary residual readout requires source A/B eligibility")
                identity = _native_episode_identity(current, registry,
                    config["paper_grade_full_suite"]["native_rollout_evaluation"], config)
                if identity["family_uid"] != uid:
                    raise ValueError("Declared independent group/episode mismatch")
                _native_validity_mask(current, registry["grid"])
                source = next(row for row in registry["sources"] if row["source_uid"] == identity["source_uid"])
                source_identity = _source_identity(source, policy)
                if source_identity not in source_ids:
                    source_ids.append(source_identity)
                metadata.append({"identity": identity, "masks": [row["teacher"]["validity_mask"] for row in current],
                    "teacher_provenance": [{key: row["teacher"][key] for key in
                        ("trajectory_uid", "source_snapshot_ref", "qc_grade", "qc_use", "uncertainty_origin",
                         "split_name", "source_holdout_role")} for row in current]})
                episodes.append(current)
                identities.append(identity)
        roles[operation] = {"episodes": episodes, "identities": identities, "metadata": metadata}
    roles["source_identities"] = source_ids
    return roles


def prediction_binding(config, checkpoint_path, loaded, grid) -> dict:
    if not native_transolver_mean_config(config):
        raise ValueError("Transolver residual binding requires its native direct predictor")
    return {"law": LAW, "direct_checkpoint_sha256": file_sha256(checkpoint_path),
        "direct_training_seed_binding": loaded["predictor_binding"],
        "direct_model_identity": direct_model_identity(config),
        "conditioning_identity": loaded["conditioning"]["identity"],
        "normalization_id": loaded["model"].field_rollout_operator.field_normalization.identity,
        "domain": source_domain(grid)}


def load_residual_artifact(path: Path, config: dict, expected_binding: dict, *, partition=None) -> dict:
    policy = residual_policy(config)
    if not path.is_file() or not policy["frozen_artifact_sha256"]:
        raise FileNotFoundError("Requested frozen residual law or its prespecified identity is missing")
    digest = file_sha256(path)
    if digest != policy["frozen_artifact_sha256"]:
        raise ValueError("Frozen residual artifact SHA256 mismatch")
    artifact = json.loads(path.read_text())
    schema = config["schemas"]["native_transolver_residual"]
    if (not set(schema["required_fields"]) <= set(artifact) or artifact["schema_version"] != schema["schema_version"]
            or artifact["status"] != "FIT_COMPLETE" or artifact["binding"] != expected_binding
            or artifact["law"] != policy["law"] or artifact["reference_time_hours"] != policy["reference_time_hours"]
            or artifact["numerical"] != policy["numerical"]
            or artifact["partition"] != {key: policy[key] for key in ("fit_groups", "check_groups")}):
        raise ValueError("Residual artifact schema, checkpoint, source domain, partition or protocol mismatch")
    scales = torch.as_tensor(artifact["scales"], dtype=torch.float64)
    counts, diagnostics = artifact["counts"], artifact["diagnostics"]
    eigenvalues = torch.as_tensor(diagnostics["hessian_eigenvalues"], dtype=torch.float64)
    if eigenvalues.shape != (SPECIES_COUNT,) or not torch.isfinite(eigenvalues).all():
        raise ValueError("Residual artifact requires exactly 15 finite Hessian eigenvalues")
    numerical = policy["numerical"]
    threshold = max(numerical["hessian_absolute_threshold"],
                    numerical["hessian_relative_threshold"]*float(eigenvalues.max()))
    if (scales.shape != (SPECIES_COUNT,) or not torch.isfinite(scales).all() or not (scales > 0).all()
            or len(counts["per_species_future_coordinate_count"]) != SPECIES_COUNT
            or not all(counts["per_species_future_coordinate_count"])
            or counts["eligible_group_count"] != counts["declared_group_count"]
            or not 0 <= diagnostics["gradient_max_abs"] <= numerical["gradient_tolerance"]
            or not float(eigenvalues.min()) > threshold
            or diagnostics["hessian_required_minimum"] != threshold
            or not torch.isfinite(torch.tensor(diagnostics["objective"],dtype=torch.float64))):
        raise ValueError("Residual artifact lacks a complete converged positive identifiable law")
    if partition is not None and (artifact["admission"] != {
            "fit": partition["fit"]["metadata"], "check": partition["check"]["metadata"],
            "source_identities": partition["source_identities"]}):
        raise ValueError("Frozen source/group/mask admission mismatch")
    return artifact


def conditional_fields(prediction: dict) -> tuple[torch.Tensor, torch.Tensor]:
    if "particle_conditional_means" in prediction:
        return prediction["particle_conditional_means"][0].double(), prediction["particle_weights"][0].double()
    return prediction["field_trajectory"][0].double().unsqueeze(0), torch.ones(1, dtype=torch.float64)


def predict_fitted_residual(prediction: dict, prepared: dict, sampler, normalization, grid,
                            artifact: dict, policy: dict, *, episode_uid: str, draw_ids=None) -> dict:
    if prediction.get("predictor_kind") == "linear_dynamics_direct_mean":
        raise ValueError("Transolver residual law cannot authorize linear dynamics")
    means, weights = conditional_fields(prediction)
    draw_ids = list(range(sampler.inference_sample_count)) if draw_ids is None else list(draw_ids)
    if not draw_ids or len(draw_ids) < 2:
        raise ValueError("Fitted law requires at least two genuine complete draws")
    posterior = prepared.get("joint_posterior")
    indices = (posterior_draw_indices(posterior, sampler, max(draw_ids)+1)[0, draw_ids]
               if posterior is not None else torch.zeros(len(draw_ids), dtype=torch.long))
    normalized = normalization.normalize(means)
    draws = residual_draws(normalized, artifact["scales"], grid["time_hours"],
        artifact["reference_time_hours"], indices, episode_uid=episode_uid,
        residual_seed=policy["residual_seed"], draw_ids=draw_ids)
    physical = normalization.inverse(draws)
    physical[:, :, 0] = means[indices, :, 0]  # preserve source t0 bit-for-bit
    prediction.update(field_trajectory_samples=physical.unsqueeze(1), field_trajectory=physical.mean(0).unsqueeze(0),
        particle_draw_indices=indices.unsqueeze(0), particle_weights=weights.unsqueeze(0),
        prediction_kind="fitted_teacher_readout_law", residual_likelihood_status="fitted",
        prediction_law=LAW, predictive_law_draw_count=len(draw_ids),
        residual_rng={"seed": policy["residual_seed"], "episode_uid": episode_uid, "draw_ids": draw_ids,
                      "stream": "sha256_law_seed_episode_draw_then_complete_time_increments"},
        physical_state_certification="not_certified_unclipped_teacher_readout",
        raw_negative_value_count=int((physical < 0).sum()))
    return prediction


def run_calibration(config: dict, checkpoint_path: Path, operation: str, artifact_path: Path) -> tuple[dict, list[Path]]:
    started = time.perf_counter()
    payload = {"status": "FIT_FAILED" if operation == "fit" else "CHECK_FAILED",
               "metrics": {}, "source_status": "calibration", "scientific_twin_status": "NOT_RUN"}
    paths = []
    try:
        policy = residual_policy(config)  # freeze numerical/selection settings before targets
        registry = native.load_registry(config)
        partition = admit_partition(config, registry, policy)
        loaded = load_direct_predictor(config, checkpoint_path,
            config["paper_grade_full_suite"]["native_rollout_evaluation"]["regimen_uids"])
        paths = [native.registry_path(config), checkpoint_path, *loaded["paths"]]
        model, predictor = loaded["model"], loaded["predictor"]
        normalizer = model.field_rollout_operator.field_normalization
        normalizer.assert_binding(registry["grid"])
        binding = prediction_binding(config, checkpoint_path, loaded, registry["grid"])
        artifact = None
        if operation == "check":
            artifact = load_residual_artifact(artifact_path, config, binding, partition=partition)
            paths.append(artifact_path)
        elif artifact_path.exists():
            raise ValueError("A frozen residual artifact cannot be overwritten")
        from Libs.exp.evaluation.evaluate_screening_system import prepare_native_episode, _native_validity_mask
        statistics, diagnostics, target_records, scored = [], [], [], []
        score_policy = dict(config["paper_grade_full_suite"]["native_rollout_evaluation"],
                            predictor_kind="transolver_fitted_residual")
        sampler = model.field_rollout_operator.quantized_mddm_rollout_operator
        protocol = metric_protocol(score_policy, sampler.quantile_levels, normalizer.identity)
        cost = {"source_wall_seconds": 0., "conditioning_wall_seconds": 0., "direct_mean_wall_seconds": 0.,
                "checkpoint_and_admission_wall_seconds": time.perf_counter()-started,
                **loaded["load_cost"], "residual_inference_wall_seconds": 0.}
        for current, identity in zip(partition[operation]["episodes"], partition[operation]["identities"]):
            tick = time.perf_counter()
            arrays, bindings = native.load_records(registry, current, config, include_teacher=False)
            paths.extend(Path(path) for path in bindings)
            mask = _native_validity_mask(current, registry["grid"])
            cost["source_wall_seconds"] += time.perf_counter()-tick
            with torch.no_grad():
                tick = time.perf_counter()
                _, prepared = prepare_native_episode(model, current, config, arrays)
                cost["conditioning_wall_seconds"] += time.perf_counter()-tick
                tick = time.perf_counter()
                prediction = predict_direct_mean(predictor, prepared, model.field_rollout_operator.quantized_mddm_rollout_operator)
                means, weights = conditional_fields(prediction)
                cost["direct_mean_wall_seconds"] += time.perf_counter()-tick
                if operation == "check":
                    tick = time.perf_counter()
                    prediction = predict_fitted_residual(prediction, prepared, sampler, normalizer,
                        registry["grid"], artifact, policy, episode_uid=identity["episode_uid"])
                    cost["residual_inference_wall_seconds"] += time.perf_counter()-tick
            tick = time.perf_counter()
            targets, bindings = native.load_records(registry, current, config)
            paths.extend(Path(path) for path in bindings)
            target_records.extend({key: row["teacher"][key] for key in
                ("trajectory_uid", "content_hash", "trajectory_store_path", "trajectory_index")} for row in current)
            cost["source_wall_seconds"] += time.perf_counter()-tick
            mu, target = normalizer.normalize(means), normalizer.normalize(torch.from_numpy(targets["trajectory"]).double())
            statistics.append(episode_statistics(mu, weights, target, mask, registry["grid"]["time_hours"],
                policy["reference_time_hours"], identity["episode_uid"], identity["family_uid"]))
            if operation == "check":
                scored.append(score_native_episode(prediction["field_trajectory_samples"][:,0],
                    torch.from_numpy(targets["trajectory"]), mask, identity, normalizer, registry["grid"], protocol))
                diagnostics.append({"identity": identity, **check_diagnostics(mu, weights, target, mask,
                    registry["grid"]["time_hours"], policy["reference_time_hours"], artifact["scales"],
                    model.field_rollout_operator.node_adjacency, registry["grid"]["species_id"])})
        groups = [row["group_uid"] for row in policy[operation+"_groups"]]
        if operation == "fit":
            fit = fit_scales(statistics, groups, policy["numerical"])
            payload.update(fit)
            if fit["status"] == "FIT_COMPLETE":
                artifact = {"schema_version": config["schemas"]["native_transolver_residual"]["schema_version"],
                    **fit, "law": LAW, "reference_time_hours": policy["reference_time_hours"],
                    "binding": binding, "numerical": policy["numerical"],
                    "partition": {key: policy[key] for key in ("fit_groups", "check_groups")},
                    "admission": {"fit": partition["fit"]["metadata"], "check": partition["check"]["metadata"],
                                  "source_identities": partition["source_identities"]},
                    "fit_target_records": target_records, "cost": {**cost, **fit["cost"]},
                    "scientific_twin_status": "NOT_RUN"}
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                with artifact_path.open("x") as handle:
                    json.dump(artifact, handle, sort_keys=True, indent=2, allow_nan=False)
                payload.update(artifact_path=str(artifact_path), artifact_sha256=file_sha256(artifact_path))
                paths.append(artifact_path)
            payload["cost"] = {**cost, **fit["cost"]}
            payload["training_seed_binding"] = binding["direct_training_seed_binding"]
        else:
            counts = information_counts(statistics, groups)
            complete = (counts["eligible_group_count"] == counts["declared_group_count"]
                        and all(counts["per_species_future_coordinate_count"]))
            scores = build_native_metric_payload(scored, protocol,
                {"source_role": "calibration", "group_partition": "calibration-check",
                 "residual_artifact_sha256": file_sha256(artifact_path)}, config)
            scores["source_status"] = "native_calibration_check_prediction"
            payload.update(status="CHECK_COMPLETE_DIAGNOSTICS_ONLY" if complete and scores["status"] == "complete" else "CHECK_INCOMPLETE",
                predictive_scores=scores,
                counts=counts, episodes=diagnostics, scientific_check_pass=None,
                objective=float(residual_objective(torch.tensor(artifact["scales"],dtype=torch.float64).log(),statistics))
                    if counts["eligible_group_count"] else None,
                artifact_sha256=file_sha256(artifact_path), frozen_fit_counts=artifact["counts"],
                frozen_fit_cost=artifact["cost"], cost={**cost, "check_wall_seconds": time.perf_counter()-started})
    except FileNotFoundError as error:
        payload.update(status="NOT_FIT_MISSING_INPUTS" if operation == "fit" else "CHECK_MISSING_INPUTS", reason=str(error))
    except (ValueError, TypeError, KeyError, RuntimeError, OSError) as error:
        payload.update(reason=str(error))
    return payload, sorted(set(paths), key=str)
