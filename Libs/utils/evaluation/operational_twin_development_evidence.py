"""Open development/calibration evidence utilities for the Operational Twin.

The functions in this module never open a sealed test partition.  They operate
on retained open-world predictions and treat the counterfactual latent-world
pair as the independent decision unit.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math
import random
import statistics


DEVELOPMENT_EVIDENCE_SCHEMA_VERSION = (
    "neuronsifter-operational-twin-development-evidence-v1"
)
PRIMARY_BASELINE_METHODS = (
    "persistence_no_action",
    "same_backbone_no_action",
)
FULL_METHOD_ID = "full_uncertainty_aware_action_conditioned"
LOWER_IS_BETTER_METRICS = (
    "action_contrast_error",
    "intervention_ranking_regret",
    "trajectory_rmse",
    "effect_direction_error",
)


def _finite_paths(value: object, path: str = "root") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            paths.extend(_finite_paths(child, f"{path}.{key}"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            paths.extend(_finite_paths(child, f"{path}[{index}]"))
    elif isinstance(value, float) and not math.isfinite(value):
        paths.append(path)
    return paths


def validate_assignment_development_evidence(
    *,
    assignment: Mapping[str, object],
    summary: Mapping[str, object],
    development_world_rows: Sequence[Mapping[str, object]],
    development_pair_rows: Sequence[Mapping[str, object]],
    calibration_world_rows: Sequence[Mapping[str, object]],
    calibration_pair_rows: Sequence[Mapping[str, object]],
    support_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, str]]:
    """Validate one assignment's derived open-world evaluation package."""

    blockers: list[dict[str, str]] = []
    assignment_uid = str(assignment["assignment_uid"])
    method_id = str(assignment["method_id"])
    seed = int(assignment["seed"])
    required_identity = {
        "assignment_uid": assignment_uid,
        "method_id": method_id,
        "seed": seed,
    }
    if summary.get("schema_version") != DEVELOPMENT_EVIDENCE_SCHEMA_VERSION or summary.get("status") != "PASS":
        blockers.append({"code": "evaluation_summary_invalid", "message": f"{assignment_uid} evaluation summary is not PASS."})
    for field, expected in required_identity.items():
        if summary.get(field) != expected:
            blockers.append({"code": "evaluation_identity_mismatch", "message": f"{assignment_uid} evaluation disagrees on {field}."})
    if summary.get("development_only") is not True or summary.get("calibration_world_only") is not True:
        blockers.append({"code": "evaluation_role_boundary_invalid", "message": f"{assignment_uid} evaluation is not open dev/cal only."})
    if summary.get("formal_qualification") is not False or summary.get("sufficient_training") is not False or summary.get("twin_qualification_pass") is not False:
        blockers.append({"code": "evaluation_claim_gate_open", "message": f"{assignment_uid} evaluation opened a forbidden claim gate."})
    device = summary.get("device_provenance")
    if not isinstance(device, Mapping) or not device.get("slurm_job_id"):
        blockers.append({"code": "evaluation_slurm_identity_missing", "message": f"{assignment_uid} evaluation lacks Slurm identity."})
    elif str(device.get("hostname", "")).split(".", 1)[0] == "mgmt2":
        blockers.append({"code": "evaluation_on_control_plane", "message": f"{assignment_uid} evaluation ran on mgmt2."})
    else:
        resource_profile = str(assignment.get("resource_profile", ""))
        if resource_profile == "gpu_single":
            allocated = any(
                str(device.get(field, "")).strip() not in {"", "0", "(null)", "N/A", "None"}
                for field in ("slurm_job_gpus", "slurm_gpus_on_node", "slurm_step_gpus")
            )
            if device.get("device_type") != "cuda" or device.get("cuda_available") is not True or not allocated:
                blockers.append({"code": "evaluation_gpu_evidence_invalid", "message": f"{assignment_uid} evaluation lacks allocated CUDA evidence."})
        elif resource_profile == "cpu_light" and device.get("device_type") != "cpu":
            blockers.append({"code": "evaluation_cpu_evidence_invalid", "message": f"{assignment_uid} CPU evaluation used a non-CPU device."})
    expected_counts = {
        "development": (144, 72),
        "calibration": (72, 36),
    }
    for role, world_rows, pair_rows in (
        ("development", development_world_rows, development_pair_rows),
        ("calibration", calibration_world_rows, calibration_pair_rows),
    ):
        world_count, pair_count = expected_counts[role]
        if len(world_rows) != world_count or len(pair_rows) != pair_count:
            blockers.append({"code": "evaluation_cardinality_mismatch", "message": f"{assignment_uid} {role} evaluation has {len(world_rows)} worlds and {len(pair_rows)} pairs."})
        for row in [*world_rows, *pair_rows]:
            for field, expected in required_identity.items():
                if row.get(field) != expected:
                    blockers.append({"code": "evaluation_row_identity_mismatch", "message": f"{assignment_uid} {role} row disagrees on {field}."})
                    break
        non_finite = _finite_paths([*world_rows, *pair_rows], f"{assignment_uid}.{role}")
        if non_finite:
            blockers.append({"code": "evaluation_non_finite", "message": f"{assignment_uid} has non-finite {role} metrics at {non_finite[:5]}."})
    calibration_eligible = bool(dict(assignment["runtime_contract"])["calibration_eligible"])
    for row in [*development_world_rows, *calibration_world_rows]:
        values = (row.get("gaussian_nll"), row.get("coverage_95"), row.get("predictive_scale_mean"))
        if calibration_eligible and any(value is None for value in values):
            blockers.append({"code": "uq_metric_missing", "message": f"{assignment_uid} lacks an uncertainty metric."})
            break
        if not calibration_eligible and any(value is not None for value in values):
            blockers.append({"code": "non_uq_metric_not_null", "message": f"{assignment_uid} fabricated an uncertainty metric."})
            break
    if len(support_rows) != 9 or any(row.get("status_match") is not True for row in support_rows):
        blockers.append({"code": "support_evidence_invalid", "message": f"{assignment_uid} support classification evidence is incomplete."})
    if any(row.get("abstention_compliant") is not True for row in support_rows):
        blockers.append({"code": "abstention_evidence_invalid", "message": f"{assignment_uid} support-driven abstention is inconsistent."})
    return blockers


def training_budget_evidence(
    status: Mapping[str, object],
    expected_budget: Mapping[str, object],
    development_metric_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Reconcile full-budget execution with development checkpoint selection.

    Historical Round 37 statuses recorded the selected best-development epoch
    in ``epochs_completed``.  The append-only development metric series is the
    authoritative record of epochs actually executed.  This function keeps
    those concepts separate and accepts a run only when either the frozen epoch
    budget or the frozen max-step budget was exhausted.
    """

    expected_epochs = int(expected_budget.get("epochs", 0))
    expected_max_steps = int(expected_budget.get("max_steps", 0))
    reported_epochs = int(status.get("epochs_completed", -1))
    steps_completed = int(status.get("steps_completed", -1))
    metric_epochs = [
        int(row["epoch"])
        for row in development_metric_rows
        if isinstance(row.get("epoch"), (int, float))
    ]
    observed_epochs = max(metric_epochs) if metric_epochs else reported_epochs
    selected_checkpoint_epoch = int(
        status.get("selected_checkpoint_epoch", reported_epochs)
    )

    if expected_epochs == 0:
        complete = observed_epochs == 0 and steps_completed == 0
        completion_mode = "no_fit" if complete else "invalid_no_fit_execution"
    else:
        epoch_complete = observed_epochs == expected_epochs
        step_complete = expected_max_steps > 0 and steps_completed == expected_max_steps
        complete = bool(epoch_complete or step_complete)
        completion_mode = (
            "epoch_budget"
            if epoch_complete
            else "max_step_budget"
            if step_complete
            else "incomplete"
        )

    return {
        "expected_epochs": expected_epochs,
        "expected_max_steps": expected_max_steps,
        "reported_epochs_completed": reported_epochs,
        "observed_epochs_completed": observed_epochs,
        "selected_checkpoint_epoch": selected_checkpoint_epoch,
        "steps_completed": steps_completed,
        "completion_mode": completion_mode,
        "full_budget_completed": complete,
    }


def summarize_counterfactual_pairs(
    world_rows: Sequence[Mapping[str, object]],
    pair_authority: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Build one action-ranking record per approved latent-world pair."""

    by_world = {str(row["world_uid"]): row for row in world_rows}
    results: list[dict[str, object]] = []
    for pair in pair_authority:
        if pair.get("approved_pair") is not True:
            raise ValueError(
                f"Counterfactual pair {pair.get('counterfactual_pair_uid')} is not approved."
            )
        member_uids = [str(value) for value in pair.get("member_world_uids", [])]
        if len(member_uids) != 2 or any(uid not in by_world for uid in member_uids):
            raise ValueError(
                f"Counterfactual pair {pair.get('counterfactual_pair_uid')} is incomplete."
            )
        members = [by_world[uid] for uid in member_uids]
        reference = [float(row["reference_utility"]) for row in members]
        predicted = [float(row["predicted_utility"]) for row in members]
        if any(not math.isfinite(value) for value in reference + predicted):
            raise ValueError(
                f"Counterfactual pair {pair.get('counterfactual_pair_uid')} has non-finite utility."
            )
        reference_best = max(range(2), key=lambda index: (reference[index], -index))
        predicted_best = max(range(2), key=lambda index: (predicted[index], -index))
        reference_contrast = reference[1] - reference[0]
        predicted_contrast = predicted[1] - predicted[0]
        if reference_contrast == 0.0:
            direction_correct: bool | None = None
            direction_error: float | None = None
        else:
            direction_correct = (reference_contrast > 0.0) == (predicted_contrast > 0.0)
            direction_error = 0.0 if direction_correct else 1.0
        results.append(
            {
                "schema_version": DEVELOPMENT_EVIDENCE_SCHEMA_VERSION,
                "role": str(members[0]["role"]),
                "assignment_uid": str(members[0]["assignment_uid"]),
                "method_id": str(members[0]["method_id"]),
                "seed": int(members[0]["seed"]),
                "counterfactual_pair_uid": str(pair["counterfactual_pair_uid"]),
                "latent_world_uid": str(pair["latent_world_uid"]),
                "family_uid": str(pair["family_uid"]),
                "member_world_uids": member_uids,
                "reference_best_world_uid": member_uids[reference_best],
                "predicted_best_world_uid": member_uids[predicted_best],
                "reference_utility_contrast": reference_contrast,
                "predicted_utility_contrast": predicted_contrast,
                "action_contrast_error": abs(predicted_contrast - reference_contrast),
                "intervention_ranking_regret": max(reference) - reference[predicted_best],
                "effect_direction_correct": direction_correct,
                "effect_direction_error": direction_error,
                "trajectory_rmse": statistics.fmean(
                    float(row["trajectory_rmse"]) for row in members
                ),
            }
        )
    return results


def _stratified_mean(rows: Sequence[Mapping[str, object]], value_key: str) -> float:
    by_family: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(value_key)
        if value is None:
            continue
        by_family[str(row["family_uid"])].append(float(value))
    family_means = [statistics.fmean(values) for values in by_family.values() if values]
    return statistics.fmean(family_means) if family_means else 0.0


def family_stratified_paired_bootstrap(
    rows: Sequence[Mapping[str, object]],
    *,
    value_key: str,
    resamples: int,
    seed: int,
    confidence_level: float,
) -> dict[str, object]:
    """Bootstrap paired method differences within family, weighting families equally."""

    if resamples <= 0 or not 0.0 < confidence_level < 1.0:
        raise ValueError("Bootstrap policy must use positive resamples and confidence in (0, 1).")
    by_family: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row.get(value_key) is not None:
            by_family[str(row["family_uid"])].append(dict(row))
    if not by_family or any(not values for values in by_family.values()):
        raise ValueError("Family-stratified bootstrap requires at least one defined row per family.")
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(resamples):
        sampled: list[dict[str, object]] = []
        for family in sorted(by_family):
            family_rows = by_family[family]
            sampled.extend(family_rows[rng.randrange(len(family_rows))] for _ in family_rows)
        draws.append(_stratified_mean(sampled, value_key))
    draws.sort()
    alpha = 1.0 - confidence_level
    lower_index = max(0, int((alpha / 2.0) * (len(draws) - 1)))
    upper_index = min(
        len(draws) - 1,
        int((1.0 - alpha / 2.0) * (len(draws) - 1)),
    )
    return {
        "method": "family_stratified_paired_bootstrap",
        "independent_unit": "latent_world",
        "resamples": resamples,
        "seed": seed,
        "confidence_level": confidence_level,
        "estimate": _stratified_mean(rows, value_key),
        "lower": draws[lower_index],
        "upper": draws[upper_index],
    }


def family_stratified_sign_permutation(
    rows: Sequence[Mapping[str, object]],
    *,
    value_key: str,
    permutations: int,
    seed: int,
) -> dict[str, object]:
    """Two-sided paired sign permutation over latent-world differences."""

    defined = [dict(row) for row in rows if row.get(value_key) is not None]
    if not defined or permutations <= 0:
        raise ValueError("Conditional permutation requires defined rows and positive permutations.")
    observed = abs(_stratified_mean(defined, value_key))
    rng = random.Random(seed)
    exceedances = 0
    for _ in range(permutations):
        permuted = [
            {**row, value_key: float(row[value_key]) * (-1.0 if rng.random() < 0.5 else 1.0)}
            for row in defined
        ]
        if abs(_stratified_mean(permuted, value_key)) >= observed:
            exceedances += 1
    return {
        "method": "family_stratified_paired_sign_permutation",
        "independent_unit": "latent_world",
        "permutations": permutations,
        "seed": seed,
        "two_sided_p_value": (exceedances + 1.0) / (permutations + 1.0),
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return Holm family-wise adjusted p-values."""

    ordered = sorted((float(value), key) for key, value in p_values.items())
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (value, key) in enumerate(ordered):
        candidate = min(1.0, (count - rank) * value)
        running = max(running, candidate)
        adjusted[key] = running
    return adjusted


def build_development_comparisons(
    pair_rows: Sequence[Mapping[str, object]],
    statistics_policy: Mapping[str, object],
) -> dict[str, object]:
    """Compare the full Twin with preregistered action-agnostic baselines."""

    keyed: dict[tuple[str, int, str], Mapping[str, object]] = {}
    for row in pair_rows:
        keyed[(str(row["method_id"]), int(row["seed"]), str(row["counterfactual_pair_uid"]))] = row
    full_keys = {
        (seed, pair_uid)
        for method_id, seed, pair_uid in keyed
        if method_id == FULL_METHOD_ID
    }
    bootstrap = dict(statistics_policy["bootstrap"])
    permutation = dict(statistics_policy["conditional_permutation"])
    comparisons: list[dict[str, object]] = []
    raw_p_values: dict[str, float] = {}
    for baseline_method in PRIMARY_BASELINE_METHODS:
        shared = sorted(
            key for key in full_keys if (baseline_method, key[0], key[1]) in keyed
        )
        by_pair: dict[str, list[dict[str, object]]] = defaultdict(list)
        for seed, pair_uid in shared:
            full = keyed[(FULL_METHOD_ID, seed, pair_uid)]
            baseline = keyed[(baseline_method, seed, pair_uid)]
            for metric in LOWER_IS_BETTER_METRICS:
                left = baseline.get(metric)
                right = full.get(metric)
                if left is None or right is None:
                    continue
                by_pair[f"{pair_uid}|{metric}"].append(
                    {
                        "family_uid": str(full["family_uid"]),
                        "metric": metric,
                        "difference": float(left) - float(right),
                    }
                )
        for metric in LOWER_IS_BETTER_METRICS:
            world_rows = []
            for pair_metric, seed_rows in by_pair.items():
                pair_uid, row_metric = pair_metric.rsplit("|", 1)
                if row_metric != metric or not seed_rows:
                    continue
                world_rows.append(
                    {
                        "counterfactual_pair_uid": pair_uid,
                        "family_uid": seed_rows[0]["family_uid"],
                        "improvement": statistics.fmean(
                            float(row["difference"]) for row in seed_rows
                        ),
                        "seed_count": len(seed_rows),
                    }
                )
            comparison_uid = f"{FULL_METHOD_ID}__vs__{baseline_method}__{metric}"
            if not world_rows:
                comparisons.append(
                    {
                        "comparison_uid": comparison_uid,
                        "full_method_id": FULL_METHOD_ID,
                        "baseline_method_id": baseline_method,
                        "metric": metric,
                        "status": "UNDEFINED",
                        "world_count": 0,
                    }
                )
                continue
            ci = family_stratified_paired_bootstrap(
                world_rows,
                value_key="improvement",
                resamples=int(bootstrap["resamples"]),
                seed=int(statistics_policy.get("analysis_seed", 370037)),
                confidence_level=float(bootstrap["confidence_level"]),
            )
            perm = family_stratified_sign_permutation(
                world_rows,
                value_key="improvement",
                permutations=int(permutation["permutations"]),
                seed=int(statistics_policy.get("analysis_seed", 370037)) + 1,
            )
            raw_p_values[comparison_uid] = float(perm["two_sided_p_value"])
            comparisons.append(
                {
                    "comparison_uid": comparison_uid,
                    "full_method_id": FULL_METHOD_ID,
                    "baseline_method_id": baseline_method,
                    "metric": metric,
                    "status": "DEFINED",
                    "world_count": len(world_rows),
                    "seed_count_per_world": sorted(
                        {int(row["seed_count"]) for row in world_rows}
                    ),
                    "improvement_definition": "baseline_error_minus_full_error",
                    "family_stratified_bootstrap_ci95": ci,
                    "conditional_permutation": perm,
                }
            )
    adjusted = holm_adjust(raw_p_values)
    for comparison in comparisons:
        uid = str(comparison["comparison_uid"])
        if uid in adjusted:
            comparison["holm_adjusted_p_value"] = adjusted[uid]
    return {
        "schema_version": DEVELOPMENT_EVIDENCE_SCHEMA_VERSION,
        "development_only": True,
        "formal_qualification": False,
        "full_method_id": FULL_METHOD_ID,
        "primary_baseline_methods": list(PRIMARY_BASELINE_METHODS),
        "comparisons": comparisons,
        "multiplicity": {
            "method": "holm",
            "family_size": len(raw_p_values),
        },
    }
