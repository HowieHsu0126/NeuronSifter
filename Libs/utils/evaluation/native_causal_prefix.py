"""Declared paired diagnostics of the joint native prefix law."""
from __future__ import annotations

import copy
import json
import math

import torch

from Libs.utils.data import native_diffusion_data as native
from Libs.models.modules.world_state import JointParticleSupport, _posterior_uid
from Libs.utils.data.native_observable_profile import quantity_support
from Libs.utils.data.native_stimulation import FORCING_ARRAY_NAMES
from Libs.utils.evaluation.native_rollout_metrics import metric_policy

PREFIX_SPEC = {
    "estimator": "off_diagonal_paired_energy_u",
    "distance": "joint_prefix_l2_over_sqrt_valid_coordinates",
    "prefix": "strictly_before_future_knot_exclude_initial",
    "mask": "exact_common_declared_prefix",
    "monte_carlo_uncertainty": "delete_one_paired_draw_jackknife",
    "aggregation": "contrasts_within_episode_then_equal_episode",
    "dtype": "float64",
    "causal_decision_rule_status": "unbound",
}
COUPLING = {
    "whole_particle": "shared_across_actions_and_times",
    "conditional_rng": "sample_regimen_draw_keyed",
    "same_conditional_noise_across_regimens": False,
    "paired_path_rms_role": "descriptive_not_a_causal_failure_rule",
}


def json_value(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def causal_prefix_policy(config: dict) -> dict:
    base = metric_policy(config)
    backbone = config["screening_pipeline"]["screening_benchmark"]["model_architecture"]["rollout_operator"]["transition_backbone"]
    if (base["predictor_kind"] != "native_diffusion" or backbone["kind"] != "neuronsifter_world_action"
            or native.native_action_policy(config) != "occupancy_program" or not native.native_enabled(config)):
        raise ValueError("Causal prefix requires the native stochastic occupancy-conditioned mainline")
    if base["source_role"] != "development":
        raise ValueError("Causal prefix requires declared development sources")
    for key in ("sample_uids", "regimen_uids"):
        values = base[key]
        if not values or not all(isinstance(x, str) and x for x in values) or len(set(values)) != len(values):
            raise ValueError("Causal prefix requires unique explicit sample/action selections")
    if not base["checkpoint_scope"] or not base["checkpoint_split_name"]:
        raise ValueError("Causal prefix requires the original checkpoint scope")
    policy = copy.deepcopy(config["paper_grade_full_suite"]["native_causal_prefix_evaluation"])
    if {key: value for key, value in policy.items() if key != "contrasts"} != PREFIX_SPEC:
        raise ValueError("Unsupported causal-prefix estimator or decision convention")
    if not isinstance(policy.get("contrasts"), list) or not policy["contrasts"]:
        raise ValueError("Causal prefix requires predeclared contrasts")
    return policy


def declared_contrast(value: dict, config: dict, duplicate_ids: set[str]) -> dict:
    required = set(config["schemas"]["native_causal_prefix"]["contrast_fields"])
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Malformed declared causal-prefix contrast")
    for key in ("contrast_uid", "sample_uid", "episode_uid"):
        if not isinstance(value[key], str) or not value[key]:
            raise ValueError("Contrast identities must be nonempty strings")
    base = config["paper_grade_full_suite"]["native_rollout_evaluation"]
    actions = value["regimen_uids"]
    if (value["contrast_uid"] in duplicate_ids or value["episode_uid"] != value["sample_uid"]
            or value["sample_uid"] not in base["sample_uids"] or not isinstance(actions, list)
            or len(actions) != 2 or not all(isinstance(x, str) and x in base["regimen_uids"] for x in actions)
            or actions[0] == actions[1]):
        raise ValueError("Duplicate or mismatched declared contrast/sample/action identity")
    cutoff = value["future_knot_hours"]
    if isinstance(cutoff, bool) or not isinstance(cutoff, (int, float)) or not math.isfinite(cutoff):
        raise ValueError("Future knot must be a finite source-grid hour")
    return copy.deepcopy(value)


def prefix_selection(grid: dict, cutoff: float) -> tuple[int, torch.Tensor]:
    times = grid["time_hours"]
    if (grid["physical_time_unit"] != "hour" or len(times) < 2
            or not all(isinstance(t, (float, int)) and not isinstance(t, bool) and math.isfinite(t) for t in times)
            or any(b <= a for a, b in zip(times, times[1:])) or cutoff not in times):
        raise ValueError("Future knot must exactly match the increasing native-hour grid")
    knot = times.index(cutoff)
    prefix = torch.tensor([index > 0 and time < cutoff for index, time in enumerate(times)])
    return knot, prefix


def common_prefix_mask(masks: torch.Tensor, grid: dict, cutoff: float) -> tuple[int, torch.Tensor, int]:
    shape = (2, *(len(grid[key]) for key in ("time_hours", "region_id", "compartment_id", "species_id")))
    if not torch.is_tensor(masks) or masks.dtype != torch.bool or tuple(masks.shape) != shape:
        raise ValueError("Prefix masks require exact bool action/time/region/compartment/species axes")
    masks = masks & torch.as_tensor(quantity_support(grid), device=masks.device)
    knot, prefix = prefix_selection(grid, cutoff)
    prefix = prefix.to(masks.device)
    if not torch.equal(masks[0, prefix], masks[1, prefix]):
        raise ValueError("Declared prefix validity masks differ")
    mask = masks[0] & prefix[:, None, None, None]
    return knot, mask, int(prefix.sum())


def score_prefix_pair(samples: torch.Tensor, masks: torch.Tensor, grid: dict, cutoff: float, normalization) -> dict:
    shape = (2, *(len(grid[key]) for key in ("time_hours", "region_id", "compartment_id", "species_id")))
    if (not torch.is_tensor(samples) or samples.ndim != 6 or tuple(samples.shape[1:]) != shape
            or not torch.is_floating_point(samples) or not torch.isfinite(samples).all()):
        raise ValueError("Prefix draws require finite full-field draw/action/time/grid axes")
    normalization.assert_domain(grid)
    normalization.assert_binding(grid)
    _, mask, time_count = common_prefix_mask(masks, grid, cutoff)
    count, n = int(mask.sum()), int(samples.shape[0])
    result = {"status": "defined", "reason": None, "prefix_time_count": time_count,
              "valid_coordinate_count": count, "realized_draw_count": n,
              "energy_u_statistic": None, "mc_standard_error": None,
              "mc_uncertainty_status": "unavailable_fewer_than_three_draws",
              "paired_path_rms": None}
    if count == 0 or n < 2:
        result.update(status="undefined", reason="empty_post_initial_prefix" if count == 0 else "fewer_than_two_draws",
                      mc_uncertainty_status="unavailable_undefined_statistic")
        return result
    values = normalization.normalize(samples.double(), torch.as_tensor(quantity_support(grid), device=samples.device))
    x, y = values[:, 0, mask], values[:, 1, mask]
    scale = math.sqrt(count)
    cross = torch.cdist(x, y) / scale
    kernel = cross + cross.T - torch.cdist(x, x) / scale - torch.cdist(y, y) / scale
    kernel.fill_diagonal_(0.)
    total = kernel.sum()
    estimate = total / (n * (n - 1))
    result["energy_u_statistic"] = float(estimate)
    result["paired_path_rms"] = float((x-y).square().mean().sqrt())
    if n >= 3:
        leave_one = (total - 2*kernel.sum(1)) / ((n-1)*(n-2))
        se = ((n-1)/n * (leave_one-leave_one.mean()).square().sum()).sqrt()
        result.update(mc_standard_error=float(se), mc_uncertainty_status="available_conditional_monte_carlo")
    if not all(value is None or math.isfinite(value) for value in
               (result["energy_u_statistic"], result["mc_standard_error"], result["paired_path_rms"])):
        raise FloatingPointError("Nonfinite causal-prefix statistic")
    return result


def match_record_conditions(records: list[dict]) -> None:
    """The first-record preparation authority must actually be common."""
    key = "joint_source" if "joint_source" in records[0] else "observation"
    if any(key not in row or row[key] != records[0][key] for row in records[1:]):
        raise ValueError("Actions do not share the same declared observation/posterior source")


def match_prepared_pair(prepared: tuple, records: list[dict], pair: list[str], grid: dict,
                        cutoff: float, masks: torch.Tensor, field) -> dict:
    _, kwargs = prepared
    regimens = [row["teacher"]["regimen_uid"] for row in records]
    sample = records[0]["teacher"]["sample_uid"]
    indices = [regimens.index(uid) for uid in pair]
    ids = kwargs.get("regimen_ids")
    if (not torch.is_tensor(ids) or not torch.equal(ids.cpu(), torch.tensor([native.stable_numeric_id(uid) for uid in regimens]))
            or not torch.equal(kwargs["sample_ids"].cpu(), torch.tensor([native.stable_numeric_id(sample)]))):
        raise ValueError("Prepared sample/regimen RNG identity mismatch")
    normalization = field.field_normalization
    normalization.assert_domain(grid)
    normalization.assert_binding(grid)
    knot, mask, time_count = common_prefix_mask(masks, grid, cutoff)
    posterior = kwargs.get("joint_posterior")
    if posterior is not None:
        if kwargs.get("regimen_uids") != regimens or "diffusion_context" in kwargs:
            raise ValueError("Prepared joint action/context identity mismatch")
        posterior.assert_current_binding()
        expected_source = native.select_joint_source(records[0]["joint_source"], [sample], regimens,
                                                     action_policy="occupancy_program")
        expected_support = JointParticleSupport.from_payload(expected_source["support"])
        if (posterior.support.to_payload() != expected_support.to_payload()
                or posterior.posterior_uid != _posterior_uid(posterior.support, posterior.observation_uids,
                    posterior.parent_posterior_uid, posterior.log_weights, posterior.particle_context)
                or not torch.equal(posterior.weights, posterior.log_weights.exp().to(posterior.support.prior_weights.dtype))):
            raise ValueError("Prepared particle/snapshot/weight/posterior binding mismatch")
        normalization.assert_support(posterior.support)
        if list(posterior.support.sample_uids) != [sample]:
            raise ValueError("Prepared posterior sample mismatch")
        contexts = native.joint_particle_contexts(posterior, kwargs["particle_programs"], regimens)
        binding = {"posterior_family": "finite_joint_particles", "posterior_uid": posterior.posterior_uid,
                   "support_uid": posterior.support.support_uid, "particle_uids": posterior.support.particle_uids,
                   "particle_weights": posterior.weights, "observation_uids": posterior.observation_uids,
                   "source_snapshot_ref": posterior.support.source_snapshot_ref,
                   "local_membrane": posterior.support.local_binding_payload() if posterior.support.local_membrane is not None else None}
    else:
        context = kwargs.get("diffusion_context")
        if not isinstance(context, dict):
            raise ValueError("Missing explicit point conditioning")
        contexts = [context]
        binding = {"posterior_family": "deterministic_compatibility",
                   "posterior_uid": records[0]["teacher"]["posterior_uid"],
                   "observation": records[0]["observation"], "particle_uids": None, "particle_weights": None}
    r, c, s = (len(grid[key]) for key in ("region_id", "compartment_id", "species_id"))
    actions, times = len(regimens), len(grid["time_hours"])
    future_difference = False
    for context in contexts:
        shapes = {"initial_state": (1,actions,r,c,s),
                  "posterior_context": (1,actions,r,c,field.operator_hidden_dim),
                  **{key:(1,actions,times,r,c,1) for key in ("occupancy","support","physical_time")}}
        if grid.get("quantity_profile") is not None:
            shapes.update({key: (1, actions, times, r, c, 1) for key in FORCING_ARRAY_NAMES})
            expected = torch.as_tensor(quantity_support(grid, posterior.support.source_uid if posterior is not None else records[0]["teacher"]["source_uid"]), device=context["quantity_support"].device).expand(1, actions, -1, -1, -1, -1)
            if context["quantity_support"].dtype != torch.bool or not torch.equal(context["quantity_support"], expected):
                raise ValueError("Prepared quantity support differs from its independent profile")
        for key, shape in shapes.items():
            value = context.get(key)
            if not torch.is_tensor(value) or tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError("Prepared finite conditioning shape mismatch: "+key)
        normalization.assert_physical_time(context["physical_time"])
        for key in ("initial_state", "posterior_context"):
            if not torch.equal(context[key][:,indices[0]], context[key][:,indices[1]]):
                raise ValueError("Prepared non-action conditioning differs: "+key)
        for key in ("occupancy", "support", *(FORCING_ARRAY_NAMES if grid.get("quantity_profile") else ())):
            value = context[key]
            if key in {"occupancy", "support", "electrical_applicability", "optical_applicability"} and not ((value >= 0) & (value <= 1)).all():
                raise ValueError("Prepared program fractions outside [0,1]")
            if not torch.equal(value[:,indices[0],:knot], value[:,indices[1],:knot]):
                raise ValueError("Prepared past program differs: "+key)
            future_difference |= not torch.equal(value[:,indices[0],knot:], value[:,indices[1],knot:])
    if not future_difference:
        raise ValueError("Declared contrast has no future occupancy/support/stimulation difference")
    program_uids = [[records[index]["teacher"]["program_uid"] for index in indices]]
    if posterior is not None:
        programs = kwargs["particle_programs"]
        positions = [programs["regimen_uids"].index(uid) for uid in pair]
        program_uids = [[particle[index]["program_uid"] for index in positions] for particle in programs["records"][0]]
    binding.update(normalization_id=normalization.identity, physical_time_hours=grid["time_hours"],
                   program_uids=program_uids,
                   prefix_time_count=time_count, valid_coordinate_count=int(mask.sum()))
    return json_value(binding)


def contrast_row(index: int, declaration, configured_draw_count: int | None) -> dict:
    evidence = copy.deepcopy(declaration)
    value = evidence if isinstance(evidence, dict) else {}
    cutoff = value.get("future_knot_hours")
    if isinstance(cutoff, float) and not math.isfinite(cutoff):
        # Rejection evidence only: validation still receives the original value.
        # A tagged object preserves its meaning without admitting a numeric knot.
        value["future_knot_hours"] = {"invalid_value_type": "nonfinite_float", "value": repr(cutoff)}
    return {"declared_index": index, "declaration": evidence,
            **{key:(value.get(key) if isinstance(value.get(key),str) else None) for key in ("contrast_uid","sample_uid","episode_uid")},
            "regimen_uids":copy.deepcopy(value.get("regimen_uids")), "future_knot_hours":value.get("future_knot_hours"),
            "source_uid": None, "family_uid": None, "episode_identity": None, "status": "failed", "reason": "not_evaluated",
            "configured_draw_count": configured_draw_count, "realized_draw_count": 0,
            "prefix_time_count": None, "valid_coordinate_count": None, "prepared_binding": None,
            "energy_u_statistic": None, "mc_standard_error": None,
            "mc_uncertainty_status": "unavailable_failed_contrast", "paired_path_rms": None,
            "prediction_provenance": None}


def build_prefix_payload(rows: list[dict], episode_costs: dict, provenance: dict, config: dict) -> dict:
    episodes = []
    for sample in dict.fromkeys(row["sample_uid"] for row in rows):
        current = [row for row in rows if row["sample_uid"] == sample]
        valid = [row for row in current if row["status"] == "defined"]
        episodes.append({"episode_uid": sample, "declared_contrast_count":len(current), "defined_contrast_count":len(valid),
                         "status":"complete" if len(valid)==len(current) else "incomplete",
                         "energy_u_statistic":sum(row["energy_u_statistic"] for row in valid)/len(valid) if valid else None,
                         "paired_path_rms":sum(row["paired_path_rms"] for row in valid)/len(valid) if valid else None,
                         "cost":episode_costs.get(sample, {"native_prediction_calls":0}),
                         "mc_uncertainty_aggregation":"not_combined_across_shared_contrast_draws"})
    values = [row for row in episodes if row["defined_contrast_count"]]
    payload = {"schema_version":config["schemas"]["native_causal_prefix"]["schema_version"],
               "operation":"causal_prefix", "status":"incomplete",
               "diagnostic_status":"complete" if all(row["status"]=="defined" for row in rows) else "incomplete",
               "causal_decision_rule_status":"unbound", "protocol":dict(PREFIX_SPEC, coupling=COUPLING),
               "contrasts":rows, "episodes":episodes, "provenance":provenance,
               "metrics":{key:{"mean":sum(row[key] for row in values)/len(values) if values else None,
                               "episode_count":len(values), "declared_episode_count":len(episodes),
                               "contrast_count":sum(row["status"]=="defined" for row in rows), "declared_contrast_count":len(rows)}
                          for key in ("energy_u_statistic","paired_path_rms")},
               "counts":{"declared_contrasts":len(rows), "defined_contrasts":sum(row["status"]=="defined" for row in rows),
                         "failed_contrasts":sum(row["status"]=="failed" for row in rows),
                         "undefined_contrasts":sum(row["status"]=="undefined" for row in rows),
                         "declared_episodes":len(episodes), "numerically_defined_episodes":len(values),
                         "complete_episodes":sum(row["status"]=="complete" for row in episodes),
                         "group_count":len({row["family_uid"] for row in rows if row["family_uid"] is not None}),
                         "mc_uncertainty_available_contrasts":sum(row["mc_standard_error"] is not None for row in rows)}}
    validate_prefix_payload(payload,config)
    return payload


def validate_prefix_payload(payload: dict, config: dict) -> None:
    schema = config["schemas"]["native_causal_prefix"]
    if (payload.get("schema_version") != schema["schema_version"] or not set(schema["required_fields"]) <= set(payload)
            or payload["operation"] != "causal_prefix" or payload["status"] != "incomplete"
            or payload["causal_decision_rule_status"] != "unbound" or payload["protocol"] != dict(PREFIX_SPEC,coupling=COUPLING)):
        raise ValueError("Causal-prefix payload convention/status mismatch")
    rows, counts = payload["contrasts"], payload["counts"]
    if (not rows or any(not set(schema["row_required_fields"]) <= set(row) for row in rows)
            or [row["declared_index"] for row in rows] != list(range(len(rows)))
            or counts["declared_contrasts"] != len(rows)
            or any(counts[label+"_contrasts"] != sum(row["status"]==label for row in rows) for label in ("defined","failed","undefined"))
            or payload["diagnostic_status"] != ("complete" if all(row["status"]=="defined" for row in rows) else "incomplete")):
        raise ValueError("Causal-prefix declared/failed contrast denominator mismatch")
    for row in rows:
        valid = row["status"] == "defined"
        if row["status"] not in {"defined","failed","undefined"} or (row["energy_u_statistic"] is not None) != valid:
            raise ValueError("Causal-prefix statistic/status mismatch")
        if valid and (row["realized_draw_count"] < 2 or not row["valid_coordinate_count"]
                      or row["paired_path_rms"] is None):
            raise ValueError("Defined prefix lacks draws or valid coordinates")
        if (row["mc_standard_error"] is not None) != (valid and row["realized_draw_count"] >= 3):
            raise ValueError("Causal-prefix jackknife availability mismatch")
    episodes = payload["episodes"]
    samples = list(dict.fromkeys(row["sample_uid"] for row in rows))
    if [episode["episode_uid"] for episode in episodes] != samples:
        raise ValueError("Causal-prefix episode identity/denominator mismatch")
    for episode, sample in zip(episodes, samples):
        current = [row for row in rows if row["sample_uid"] == sample]
        valid = [row for row in current if row["status"] == "defined"]
        expected = {"declared_contrast_count": len(current), "defined_contrast_count": len(valid),
                    "status": "complete" if len(valid) == len(current) else "incomplete",
                    "mc_uncertainty_aggregation": "not_combined_across_shared_contrast_draws"}
        for key in ("energy_u_statistic", "paired_path_rms"):
            expected[key] = sum(row[key] for row in valid)/len(valid) if valid else None
        if any(episode.get(key) != value for key, value in expected.items()):
            raise ValueError("Causal-prefix within-episode reduction mismatch")
    expected_counts = {"declared_episodes": len(episodes),
        "numerically_defined_episodes": sum(episode["defined_contrast_count"] > 0 for episode in episodes),
        "complete_episodes": sum(episode["status"] == "complete" for episode in episodes),
        "group_count": len({row["family_uid"] for row in rows if row["family_uid"] is not None}),
        "mc_uncertainty_available_contrasts": sum(row["mc_standard_error"] is not None for row in rows)}
    if any(counts.get(key) != value for key, value in expected_counts.items()):
        raise ValueError("Causal-prefix episode/group/uncertainty denominator mismatch")
    if set(payload["metrics"]) != {"energy_u_statistic", "paired_path_rms"}:
        raise ValueError("Causal-prefix required descriptive metrics mismatch")
    for key, metric in payload["metrics"].items():
        values = [row[key] for row in episodes if row[key] is not None]
        mean = sum(values)/len(values) if values else None
        if metric != {"mean":mean,"episode_count":len(values),"declared_episode_count":len(episodes),
                      "contrast_count":sum(row["status"]=="defined" for row in rows),"declared_contrast_count":len(rows)}:
            raise ValueError("Causal-prefix equal-episode reduction mismatch")
    json.dumps(payload,allow_nan=False)
