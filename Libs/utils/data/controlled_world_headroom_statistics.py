"""Development-only oracle headroom statistics for controlled neural-world benchmark."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
import random
from pathlib import Path
from statistics import mean, variance
from Libs.utils.data.controlled_world_benchmark_manifests import (
    TERMINALS,
    _materialized_config,
    validate_benchmark_config,
)
from Libs.utils.data.controlled_world_contracts import (
    FAMILY_UIDS,
    ControlledWorldContractError,
    bind_world_action,
    select_family_config,
    validate_controlled_world_config,
)
from Libs.utils.data.controlled_world_runtime import (
    evaluate_harness_true_utility,
    load_harness_oracle,
    simulate_harness_trajectory,
)
from Libs.utils.data.trajectory_contracts import canonical_sha256, stable_uid

HEADROOM_SCHEMA_VERSION = "neuronsifter-controlled-world-headroom-statistics-v1"
QUALIFICATION_REPORT_SCHEMA_VERSION = "neuronsifter-controlled-world-qualification-report-v1"


def build_development_headroom(config: Mapping[str, object], development_manifest: Mapping[str, object], *, repository_root: Path) -> dict[str, object]:
    source = validate_controlled_world_config(config)
    bench = validate_benchmark_config(config)
    if development_manifest.get("role") != "development":
        raise ControlledWorldContractError("development_only_violation", "Oracle information-headroom may use development worlds only.")
    records: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    undefined_cases: list[dict[str, object]] = []
    for row in development_manifest["worlds"]:
        try:
            records.append(_headroom_record(source, bench, row, repository_root))
        except ControlledWorldContractError as exc:
            failure = {"world_uid": row.get("world_uid"), "family_uid": row.get("family_uid"), **exc.to_blocker()}
            failures.append(failure)
            undefined_cases.append({"world_uid": row.get("world_uid"), "family_uid": row.get("family_uid"), "reason": exc.code})
    stats = _statistics(records, bench)
    report = {
        "schema_version": HEADROOM_SCHEMA_VERSION,
        "headroom_uid": stable_uid("controlled_world_development_headroom", {"records": records, "statistics": stats}),
        "evidence_class": "controlled_world_non_empirical",
        "protocol": bench["headroom_policy"]["protocol"],
        "development_only": True,
        "independent_unit": "world",
        "family_uids": list(FAMILY_UIDS),
        "world_count": len(records),
        "records": records,
        "statistics": stats,
        "failure_records": failures,
        "undefined_cases": undefined_cases,
        "access_counters": {"six_context_outcome_access": 0, "protected_holdout_access": 0, "test_outcome_access": 0},
        "publication_firewall": {
            "harness_causal_state_serialized": False,
            "harness_optimum_serialized": False,
            "harness_parameters_serialized": False,
            "harness_utility_serialized": False,
        },
        "content_identity": canonical_sha256({"records": records, "statistics": stats, "failures": failures, "undefined": undefined_cases}),
    }
    _ensure_no_oracle_fields(report)
    return report


def _headroom_record(source: Mapping[str, object], bench: Mapping[str, object], world_row: Mapping[str, object], repository_root: Path) -> dict[str, object]:
    family_uid = str(world_row["family_uid"])
    index = int(str(world_row["world_uid"]).rsplit("_", 1)[-1])
    seed = int(world_row["seed"])
    materialized = _materialized_config(source, family_uid, "development", index, seed)
    family = select_family_config(materialized, family_uid)
    utilities: list[float] = []
    action_identities: list[str] = []
    for action, extension in _action_grid(family, bench):
        binding = bind_world_action(family, action, extension)
        oracle = load_harness_oracle(materialized, family_uid=family_uid, reader_role="benchmark_harness")
        trajectory = simulate_harness_trajectory(family, oracle, binding.to_dict())
        utility = evaluate_harness_true_utility(family, oracle, binding.to_dict(), trajectory, reader_role="benchmark_harness")
        utilities.append(float(utility))
        action_identities.append(canonical_sha256({"action": binding.canonical_world_action, "extension": binding.controlled_extension}))
    if not utilities or any(not math.isfinite(value) for value in utilities):
        raise ControlledWorldContractError("undefined_headroom", "Headroom utilities are undefined or non-finite.")
    best = max(utilities)
    no_info_idx = int(dict(bench["action_grid"])["no_information_action_index"])
    if no_info_idx < 0 or no_info_idx >= len(utilities):
        raise ControlledWorldContractError("invalid_headroom_policy", "No-information action index is outside the action grid.")
    regret_without_information = best - utilities[no_info_idx]
    regret_after_optimal_information = 0.0
    improvement = regret_without_information - regret_after_optimal_information
    sensitivity = max(utilities) - min(utilities)
    uncertainty_sensitivity = variance(utilities) if len(utilities) > 1 else 0.0
    return {
        "world_uid": world_row["world_uid"],
        "family_uid": family_uid,
        "latent_generator_lineage_uid": world_row["latent_generator_lineage_uid"],
        "seed_identity": world_row["seed_identity"],
        "action_grid_identity": canonical_sha256(action_identities),
        "nontrivial_optimal_action": utilities.index(best) != no_info_idx,
        "action_sensitivity": sensitivity,
        "uncertainty_sensitivity": uncertainty_sensitivity,
        "regret_without_information": regret_without_information,
        "regret_after_optimal_information": regret_after_optimal_information,
        "regret_improvement": improvement,
        "ood_support_failures": 0,
        "status": "defined",
        "access_counters": {"six_context_outcome_access": 0, "protected_holdout_access": 0, "test_outcome_access": 0},
    }


def _action_grid(family: Mapping[str, object], bench: Mapping[str, object]) -> list[tuple[dict[str, object], dict[str, object]]]:
    grid = []
    support = dict(family["support_domain"])
    bounds = dict(support["supported_intensity"])
    duration_bounds = dict(support["supported_duration"])
    fractions = list(dict(bench["action_grid"])["intensity_support_fractions"])
    duration_fraction = float(dict(bench["action_grid"])["duration_support_fraction"])
    duration = float(duration_bounds["minimum"]) + duration_fraction * (float(duration_bounds["maximum"]) - float(duration_bounds["minimum"]))
    for fraction in fractions:
        action = deepcopy(family["world_action"]["canonical"])
        extension = deepcopy(family["world_action"]["extension"])
        intensity = float(bounds["minimum"]) + float(fraction) * (float(bounds["maximum"]) - float(bounds["minimum"]))
        extension["dose_or_intensity"]["value"] = intensity
        extension["duration"]["value"] = duration
        action["duration"]["value"] = duration
        if action["intervention_type"] == "pharmacological":
            action["dose"] = intensity
        grid.append((action, extension))
    return grid


def _statistics(records: Sequence[Mapping[str, object]], bench: Mapping[str, object]) -> dict[str, object]:
    improvements = [float(row["regret_improvement"]) for row in records]
    by_family: dict[str, list[float]] = defaultdict(list)
    for row in records:
        by_family[str(row["family_uid"])].append(float(row["regret_improvement"]))
    family_means = {family_uid: (mean(values) if values else 0.0) for family_uid, values in by_family.items()}
    ci = _family_stratified_bootstrap(by_family, bench)
    pilot_variance = variance(improvements) if len(improvements) > 1 else 0.0
    sample_plan = _sample_size_plan(pilot_variance, bench)
    minimum_mean = float(dict(bench["headroom_policy"])["minimum_mean_regret_improvement"])
    min_families = int(dict(bench["headroom_policy"])["minimum_improving_family_count"])
    improving_family_count = sum(1 for value in family_means.values() if value > minimum_mean)
    return {
        "overall_mean_regret_improvement": mean(improvements) if improvements else 0.0,
        "family_mean_regret_improvement": family_means,
        "family_stratified_paired_bootstrap_ci95": ci,
        "family_spanning_improvement": improving_family_count >= min_families and all(family in family_means for family in FAMILY_UIDS),
        "improving_family_count": improving_family_count,
        "pilot_variance": pilot_variance,
        "sample_size_plan": sample_plan,
        "multiplicity_handling": dict(bench["statistical_policy"]["multiplicity"]),
        "failure_and_undefined_handling": dict(bench["statistical_policy"]["failure_and_undefined_handling"]),
    }


def _family_stratified_bootstrap(by_family: Mapping[str, Sequence[float]], bench: Mapping[str, object]) -> dict[str, object]:
    reps = int(dict(bench["headroom_policy"])["bootstrap_replicates"])
    seed = int(dict(bench["headroom_policy"])["bootstrap_seed"])
    rng = random.Random(seed)
    values: list[float] = []
    families = [family for family in FAMILY_UIDS if family in by_family and by_family[family]]
    for _ in range(reps):
        sampled = []
        for family in families:
            rows = list(by_family[family])
            sampled.extend(rows[rng.randrange(len(rows))] for _ in rows)
        values.append(mean(sampled) if sampled else 0.0)
    values.sort()
    lo_index = max(0, int(0.025 * (len(values) - 1)))
    hi_index = min(len(values) - 1, int(0.975 * (len(values) - 1)))
    return {"lower": values[lo_index], "upper": values[hi_index], "replicates": reps, "seed": seed, "method": "family_stratified_paired_bootstrap"}


def _sample_size_plan(pilot_variance: float, bench: Mapping[str, object]) -> dict[str, object]:
    policy = dict(bench["sample_size_policy"])
    effect = float(policy["minimum_detectable_regret_improvement"])
    # Conservative normal approximation for two-sided alpha=0.05 and power=0.80.
    required = math.ceil(((1.96 + 0.84) ** 2 * max(pilot_variance, 1e-12)) / max(effect**2, 1e-12))
    required = max(int(policy["minimum_worlds_per_family"]), required)
    feasible = required <= int(policy["maximum_worlds_per_family"])
    return {
        "unit": "independent_world",
        "required_worlds_per_family": required,
        "minimum_worlds_per_family": int(policy["minimum_worlds_per_family"]),
        "maximum_worlds_per_family": int(policy["maximum_worlds_per_family"]),
        "feasible": feasible,
        "alpha": float(policy["two_sided_alpha"]),
        "power": float(policy["power"]),
    }


def qualification_decision(headroom: Mapping[str, object], ood_manifest: Mapping[str, object], bench_config: Mapping[str, object]) -> dict[str, object]:
    validate_benchmark_config(bench_config)
    stats = dict(headroom["statistics"])
    terminal = TERMINALS[0]
    blockers: list[dict[str, object]] = []
    if headroom.get("failure_records"):
        blockers.append({"code": "development_failures_retained", "count": len(headroom["failure_records"])})
    if headroom.get("undefined_cases"):
        blockers.append({"code": "undefined_cases_retained", "count": len(headroom["undefined_cases"])})
    ci = dict(stats["family_stratified_paired_bootstrap_ci95"])
    if float(ci["lower"]) <= 0.0:
        blockers.append({"code": "bootstrap_ci_not_positive", "lower": ci["lower"], "upper": ci["upper"]})
    if stats.get("family_spanning_improvement") is not True:
        blockers.append({"code": "family_spanning_improvement_missing"})
    if dict(stats["sample_size_plan"]).get("feasible") is not True:
        blockers.append({"code": "sample_size_not_feasible", "sample_size_plan": stats["sample_size_plan"]})
    if sorted(ood_manifest.get("ood_classes", [])) != sorted(("parametric_ood", "intervention_ood", "combination_ood", "observation_shift_ood", "structural_ood")):
        blockers.append({"code": "ood_manifest_incomplete"})
    if blockers:
        terminal = TERMINALS[1]
    report = {
        "schema_version": QUALIFICATION_REPORT_SCHEMA_VERSION,
        "qualification_uid": stable_uid("controlled_world_qualification_report", {"headroom": headroom.get("content_identity"), "ood": ood_manifest.get("content_identity"), "blockers": blockers}),
        "terminal_decision": terminal,
        "allowed_terminals": list(TERMINALS),
        "evidence_class": "controlled_world_non_empirical",
        "decision_basis": "development_only_oracle_information_headroom_and_frozen_manifest_prerequisites",
        "blockers": blockers,
        "statistics": stats,
        "failure_records": list(headroom.get("failure_records", [])),
        "undefined_cases": list(headroom.get("undefined_cases", [])),
        "access_counters": {"six_context_outcome_access": 0, "protected_holdout_access": 0, "test_outcome_access": 0},
        "claim_boundary": {"controlled_world_evidence_only": True, "empirical_evidence": False, "therapeutic_efficacy_claim": False, "clinical_validation_claim": False},
    }
    if report["terminal_decision"] not in TERMINALS:
        raise ControlledWorldContractError("invalid_terminal_policy", "Terminal decision is not exact.")
    _ensure_no_oracle_fields(report)
    return report


def _ensure_no_oracle_fields(value: object) -> None:
    """Reject hidden/protected payloads while allowing required zero audit counters."""

    forbidden_keys = {
        "oracle_state",
        "oracle_parameters",
        "hidden_state",
        "hidden_parameters",
        "hidden_optimum",
        "true_utility",
        "future_observation",
        "future_observations",
        "protected_outcome",
        "protected_outcomes",
    }
    zero_counter_keys = {"six_context_outcome_access", "protected_holdout_access", "test_outcome_access"}
    forbidden_value_markers = (
        "oracle_state",
        "oracle_parameters",
        "hidden_state",
        "hidden_parameters",
        "hidden_optimum",
        "true_utility",
        "future_observation",
        "future_observations",
        "protected_outcome",
        "six_context_outcome",
        "protected_holdout",
    )

    def walk(item: object, path: str = "") -> None:
        if isinstance(item, Mapping):
            for raw_key, raw_value in item.items():
                key = str(raw_key)
                current_path = f"{path}.{key}" if path else key
                normalized = key.lower()
                if normalized in zero_counter_keys:
                    if raw_value != 0:
                        raise ControlledWorldContractError(
                            "protected_access_nonzero",
                            "Protected and test access counters must remain exactly zero.",
                            field=current_path,
                        )
                    continue
                if normalized in forbidden_keys or normalized.startswith("oracle_") or normalized.startswith("hidden_"):
                    raise ControlledWorldContractError(
                        "oracle_access_forbidden",
                        "Public benchmark record contains a hidden/protected field.",
                        field=current_path,
                    )
                walk(raw_value, current_path)
            return
        if isinstance(item, list):
            for index, child in enumerate(item):
                walk(child, f"{path}[{index}]")
            return
        if isinstance(item, str):
            lowered = item.lower()
            if any(marker in lowered for marker in forbidden_value_markers):
                raise ControlledWorldContractError(
                    "oracle_access_forbidden",
                    "Public benchmark record contains a hidden/protected value token.",
                    field=path,
                )

    walk(value)
