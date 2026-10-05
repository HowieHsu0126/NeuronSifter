"""Reproducible finite-law comparison of declared dependence query channels.

This studies mathematical information access, not biological acquisition cost.
Every source law has the supplied singleton marginals by construction. A summary
channel counts two scalars per paired-block observation; full-trace access gives
all statistics together and provides no acquisition advantage to local queries.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import scipy
import torch
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, vstack

from Libs.models.modules.neuronsifter_world_action import (
    GraphSpectralTemporalWaveletTokenizer,
)
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.evaluation.marginal_intervention_risk import temporal_synthesis_matrix
from Libs.utils.evaluation.haar_joint_observation import (
    compile_joint_observation,
    joint_observation_bounds,
)
from Libs.utils.evaluation.haar_observation_experiments import (
    bind_haar_observation_counts,
    categorical_kl,
    compile_haar_observation_channel,
    haar_observation_risk_bounds,
    observation_probabilities,
)
from Libs.utils.evaluation.haar_query_preserving_compression import (
    compile_query_memory,
    compile_query_preserving_haar,
    compile_query_readout,
    query_memory_bounds,
)
from Libs.utils.evaluation.multiscale_innovation_sampling import (
    InnovationLocations,
    compile_innovation_design,
    estimate_innovation_risk,
    innovation_importance_values,
    sample_innovation_locations,
)
from Libs.utils.evaluation.posterior_wavelet_risk import (
    PosteriorMomentQuery,
    bind_posterior_report,
    compile_posterior_haar,
    posterior_contrast_bounds,
    posterior_worst_report_regret,
    select_posterior_query,
)
from Libs.utils.evaluation.wavelet_dependence_queries import (
    bind_haar_moment_intervals,
    compile_haar_dependence_queries,
    condition_haar_risk,
    evaluate_haar_innovations,
    haar_energy_tail_bound,
    haar_resampling_influences,
    select_compressed_resampling_query,
    select_resampling_screened_query,
    report_from_matched_blocks,
    worst_report_decision_regret,
)


def quantile_source_law(probabilities, kind, seed):
    """Finite common-uniform construction with exactly the prescribed marginals."""
    rng = np.random.default_rng(seed)
    m, _ = probabilities.shape
    cdf = probabilities.cumsum(-1)
    shifts = rng.uniform(size=m) if kind == "shifted_quantile" else np.zeros(m)
    signs = (
        np.where(np.arange(m) % 2, -1, 1)
        if kind == "reflected_quantile"
        else np.ones(m)
    )
    edges = [0.0, 1.0]
    for i in range(m):
        edges.extend(((signs[i] * (cdf[i, :-1] - shifts[i])) % 1).tolist())
        edges.append(float((-signs[i] * shifts[i]) % 1))
    edges = np.unique(edges)
    masses = np.diff(edges)
    midpoints = (edges[1:] + edges[:-1]) / 2
    uniforms = (midpoints[:, None] * signs[None] + shifts[None]) % 1
    indices = np.stack(
        [np.searchsorted(cdf[i], uniforms[:, i], side="right") for i in range(m)],
        axis=1,
    )
    recovered = np.stack(
        [
            np.bincount(indices[:, i], weights=masses, minlength=probabilities.shape[1])
            for i in range(m)
        ]
    )
    if not np.allclose(recovered, probabilities, atol=1e-12, rtol=0):
        raise ArithmeticError("Constructed law does not preserve supplied marginals.")
    return indices, masses


def _risk_regret(interval, alternative):
    return min(max(0.0, interval[1] - alternative), max(0.0, alternative - interval[0]))


def evaluate_binary_decision(interval, alternative, true_risk):
    """Separate report-conditional worst regret from source-law decision loss.

    A tie in the two conditional regret bounds deterministically selects the
    candidate. The true risk is used only for evaluation after that choice.
    """
    candidate_bound = max(0.0, interval[1] - alternative)
    comparator_bound = max(0.0, alternative - interval[0])
    choose_candidate = candidate_bound <= comparator_bound
    chosen_risk = true_risk if choose_candidate else alternative
    return {
        "decision": "candidate" if choose_candidate else "comparator",
        "conditional_minimax_regret": min(candidate_bound, comparator_bound),
        "realized_decision_regret": max(0.0, chosen_risk - min(true_risk, alternative)),
    }


def uniform_goal_sample_range(synthesis, reference, threshold):
    """Attained range for {-1,0,1}, a balanced tree and a uniform hinge goal.

    The maximum sets root and all details to +1: equal child objectives make
    each detail sign immaterial, and convexity maximizes its magnitude. The
    minimum sets the root to -1 and details to zero (Jensen equality).
    """
    n = synthesis.shape[0]
    lower = max(0.0, reference - 1 / math.sqrt(n) - threshold)
    upper = float(
        np.maximum(0.0, reference + synthesis @ np.ones(n) - threshold).mean()
    )
    return lower, upper


def _sample_block_means(field, nodes, levels):
    columns = []
    for depth, position in nodes:
        width = 2 ** (levels - depth)
        start = position * width
        columns.append(
            np.stack(
                [
                    field[:, start : start + width // 2].mean(-1),
                    field[:, start + width // 2 : start + width].mean(-1),
                ],
                axis=-1,
            )
        )
    return np.stack(columns, axis=1)


def threshold_stable_access_family(time_count):
    """Finite support for the physical access theorem, decoded by the real codec.

    The 144 assignments exhaust the sparse marginal supports: nine coarse
    pairs and sixteen fine-sign patterns. Horizon growth refines the grid;
    it does not increase the number of active coefficient levels.
    """
    if (
        isinstance(time_count, bool)
        or not isinstance(time_count, int)
        or time_count < 8
        or time_count & (time_count - 1)
    ):
        raise ValueError("The access family needs a power-of-two horizon >= 8.")
    n = time_count
    levels = n.bit_length() - 1
    k, reference = 1 / 8, 1 / 3
    threshold = reference + k / 2
    code = np.arange(-1, 17, dtype=float) * k * math.sqrt(n) / 8
    tuples = np.array(list(itertools.product(range(3), range(3), *[(-1, 1)] * 4)))
    indices = np.ones((len(tuples), n), dtype=int)
    indices[:, :2] = 1 + 8 * tuples[:, :2]
    indices[:, 4:8] = 1 + tuples[:, 2:]
    coefficients = torch.tensor(code[indices], dtype=torch.float64)
    codec = GraphSpectralTemporalWaveletTokenizer(4, 2, levels, 1, False)
    metadata = codec.pack_temporal_wavelets(coefficients.new_zeros((1, n, 1, 1)))[1]
    field_tensor = (
        reference
        + codec.unpack_temporal_wavelets(coefficients[:, :, None, None], metadata)[
            :, :, 0, 0
        ]
    )
    field = field_tensor.numpy()
    repacked = codec.pack_temporal_wavelets(
        (field_tensor - reference)[:, :, None, None]
    )[0][:, :, 0, 0].numpy()
    nodes = tuple((depth, pos) for depth in range(levels) for pos in range(2**depth))
    blocks = _sample_block_means(field, nodes, levels)
    widths = np.array([n // 2**depth for depth, _ in nodes])
    means, contrasts = blocks.mean(-1), (blocks[:, :, 0] - blocks[:, :, 1]) / 2
    innovations = (
        widths[None]
        / (2 * n)
        * np.maximum(0, np.abs(contrasts) - np.abs(means - threshold))
    )
    identity = (tuples[:, 1] == tuples[:, 0]).astype(float) / 48
    cycle = (tuples[:, 1] == (tuples[:, 0] + 1) % 3).astype(float) / 48
    return {
        "time_count": n,
        "codebook": code,
        "indices": indices,
        "coarse_indices": tuples[:, :2],
        "field": field,
        "coefficients": coefficients.numpy(),
        "repacked": repacked,
        "half_means": blocks[:, 0],
        "innovations": innovations,
        "costs": np.maximum(0, field - threshold).mean(1),
        "root_costs": np.maximum(0, k * tuples[:, 0] - k / 2),
        "identity": identity,
        "cycle": cycle,
    }


def _finite_kl(p, q):
    support = p > 0
    if np.any(q[support] <= 0):
        return math.inf
    return float(np.sum(p[support] * np.log(p[support] / q[support])))


def evaluate_access_family(config):
    """Check support identities and information retained by competing channels."""
    epsilon, alpha = config["decision_margin"], config["failure_probability"]
    if not 0 < epsilon < 1 / 192 or not 0 < alpha < 0.5:
        raise ValueError("Invalid margin or error probability for the access theorem.")
    delta, baseline, comparator = 1 / 48, 1 / 12, 11 / 96
    u = epsilon / delta
    theta_minus, theta_plus = 0.5 - u, 0.5 + u
    expected_kl = 2 * u * math.log(theta_plus / theta_minus)
    rows = []
    for n in config["time_counts"]:
        family = threshold_stable_access_family(n)
        p_i, p_c = family["identity"], family["cycle"]
        minus = (1 - theta_minus) * p_i + theta_minus * p_c
        plus = (1 - theta_plus) * p_i + theta_plus * p_c
        indices = family["indices"]
        marginals = [
            np.stack(
                [
                    np.bincount(indices[:, j], weights=law, minlength=18)
                    for j in range(n)
                ]
            )
            for law in (minus, plus)
        ]
        # Exact coarse labels are recovered from physical summaries; check the
        # numerical residual before mapping back to those labels.
        half_means = family["half_means"]
        recovered_a = (half_means.mean(1) - 1 / 3) * 8
        recovered_d = (half_means[:, 0] - half_means[:, 1]) * 4
        recovered = np.stack([recovered_a, recovered_d], axis=1)
        coarse_error = float(np.max(np.abs(recovered - family["coarse_indices"])))
        pair_groups = (
            3 * family["coarse_indices"][:, 0] + family["coarse_indices"][:, 1]
        )
        right_groups = (
            family["coarse_indices"][:, 0] - family["coarse_indices"][:, 1] + 2
        )

        def channel_kl(groups):
            return _finite_kl(
                np.bincount(groups, weights=plus), np.bincount(groups, weights=minus)
            )

        risks = [float(law @ family["costs"]) for law in (p_i, p_c, minus, plus)]
        checks = {
            "same_singleton_marginals": np.max(np.abs(marginals[0] - marginals[1]))
            < 1e-12,
            "physical_bounds": np.min(family["field"]) >= 5 / 96 - 1e-12
            and np.max(family["field"]) <= 83 / 96 + 1e-12,
            "codec_roundtrip": np.max(
                np.abs(family["repacked"] - family["coefficients"])
            )
            < 1e-11,
            "paired_summary_reconstruction": coarse_error < 1e-11,
            "fine_innovations_zero": np.max(np.abs(family["innovations"][:, 1:]))
            < 1e-12,
            "pointwise_root_sufficiency": np.max(
                np.abs(
                    family["costs"] - family["root_costs"] - family["innovations"][:, 0]
                )
            )
            < 1e-12,
            "nonconstant_fine_field": np.all(
                np.ptp(family["field"].reshape(-1, 4, n // 4), axis=-1) > 0
            ),
            "risk_values": np.max(
                np.abs(
                    np.array(risks)
                    - [5 / 48, 1 / 8, comparator - epsilon, comparator + epsilon]
                )
            )
            < 1e-12,
            "known_root_expectation": abs(p_i @ family["root_costs"] - baseline) < 1e-12
            and abs(p_c @ family["root_costs"] - baseline) < 1e-12,
        }
        kls = {
            "full_trace": _finite_kl(plus, minus),
            "paired_halves": channel_kl(pair_groups),
            "right_half_only": channel_kl(right_groups),
        }
        checks["equal_channel_information"] = (
            max(abs(v - expected_kl) for v in kls.values()) < 1e-12
        )
        if not all(checks.values()):
            raise ArithmeticError(f"Access-family identity failed at N={n}: {checks}")
        rows.append(
            {
                "time_count": n,
                "support_assignments": len(indices),
                "active_detail_levels": [0, 2],
                "codebook_size": len(family["codebook"]),
                "physical_range": [
                    float(family["field"].min()),
                    float(family["field"].max()),
                ],
                "innovation_range": [
                    float(family["innovations"][:, 0].min()),
                    float(family["innovations"][:, 0].max()),
                ],
                "identity_risk": risks[0],
                "cycle_risk": risks[1],
                "negative_margin_risk": risks[2],
                "positive_margin_risk": risks[3],
                "per_replicate_kl": kls,
                "necessary_fixed_replicates": (1 - 2 * alpha)
                * math.log((1 - alpha) / alpha)
                / expected_kl,
                "sufficient_component_replicates": math.ceil(
                    delta**2 * math.log(2 / alpha) / (2 * epsilon**2)
                ),
                "sufficient_innovation_replicates": math.ceil(
                    (3 / 32) ** 2 * math.log(2 / alpha) / (2 * epsilon**2)
                ),
                "reported_scalars_per_replicate": {
                    "full_trace": n,
                    "paired_halves": 2,
                    "right_half_only": 1,
                },
                "checks": {key: bool(value) for key, value in checks.items()},
            }
        )
    return {
        "status": "verified_constructed_access_family",
        "config": config,
        "records": rows,
        "scope": "coefficient_marginal_nonidentifiability_and_declared_reporting_costs",
        "pairing_necessity": False,
        "biological_sample_advantage": False,
        "horizon_scaling": "finer_sampling_grid_with_two_active_detail_levels",
    }


def run_evaluation(config):
    if config["codebook_values"] != [-1.0, 0.0, 1.0]:
        raise ValueError(
            "This declared comparison uses the three-symbol symmetric alphabet."
        )
    if (
        not 0 < config["failure_probability"] < 1
        or not 0 < config["alternative_interval_fraction"] < 1
    ):
        raise ValueError("Invalid probability or comparator fraction.")
    records = []
    for n in config["time_counts"]:
        if n < 2 or n & (n - 1):
            raise ValueError(
                "The matched block experiment requires power-of-two time counts."
            )
        for seed in config["seeds"]:
            rng = np.random.default_rng(seed + 1700 * n)
            code = torch.tensor(config["codebook_values"], dtype=torch.float64)
            codec = GraphSpectralTemporalWaveletTokenizer(4, 2, 16, 1, False)
            metadata = codec.pack_temporal_wavelets(code.new_zeros((1, n, 1, 1)))[1]
            synthesis = temporal_synthesis_matrix(codec, n, code).numpy()
            spread = rng.uniform(0.05, 0.45, len(synthesis.T))
            p = np.stack([spread, 1 - 2 * spread, spread], axis=-1)
            reference = float(config["physical_reference"])
            threshold = (
                reference
                + config["goal_threshold_offsets"][
                    seed % len(config["goal_threshold_offsets"])
                ]
            )
            context = f"finite_law_time{n}_seed{seed}"
            start = time.perf_counter()
            problem = compile_haar_dependence_queries(
                torch.from_numpy(p),
                code,
                metadata,
                code.new_full((n,), reference),
                code.new_full((n,), 1 / n),
                code.new_tensor(-threshold),
                lower_bound=threshold,
                conditioning_key=context,
                max_variables=config["max_transport_variables"],
            )
            initial = condition_haar_risk(problem, None, conditioning_key=context)
            alternative = initial["lower"] + config["alternative_interval_fraction"] * (
                initial["upper"] - initial["lower"]
            )
            exact_records = []
            for statistic in ["innovation", "cross_moment", "direct_goal"]:
                count = 1 if statistic == "direct_goal" else len(problem.node_ids)
                for q in range(count):
                    result = worst_report_decision_regret(
                        problem,
                        (q,),
                        np.zeros(1),
                        alternative_loss=alternative,
                        conditioning_key=context,
                        max_variables=config["max_transport_variables"],
                        statistic=statistic,
                    )
                    exact_records.append(
                        {
                            "statistic": statistic,
                            "query_id": q,
                            "regret": result["worst_report_regret"],
                        }
                    )
            energy = p @ code.numpy() ** 2
            offsets = np.cumsum(problem.sizes)[:-1]
            query_energy = np.array(
                [energy[offsets[d] + position] for d, position in problem.node_ids]
            )
            innovation_scores = np.array(
                [r["regret"] for r in exact_records if r["statistic"] == "innovation"]
            )
            best_exact = int(innovation_scores.argmin())
            energy_query = int(query_energy.argmax())
            sampling_records = []
            for replicates in config["replicate_counts"]:
                if replicates < 1:
                    raise ValueError("replicate_counts must be positive.")
                qcount = len(problem.node_ids)
                concentration = math.sqrt(
                    math.log(2 * (2 * qcount + 1) / config["failure_probability"])
                    / (2 * replicates)
                )
                noisy_records = []
                for statistic in ["innovation", "cross_moment", "direct_goal"]:
                    if statistic == "direct_goal":
                        lower, upper = uniform_goal_sample_range(
                            synthesis, reference, threshold
                        )
                        spans = np.array([upper - lower])
                    else:
                        ranges = (
                            problem.ranges
                            if statistic == "innovation"
                            else problem.cross_moment_ranges
                        )
                        spans = np.diff(ranges, axis=1).ravel()
                    for q, span in enumerate(spans):
                        radius = float(span * concentration)
                        result = worst_report_decision_regret(
                            problem,
                            (q,),
                            np.array([radius]),
                            alternative_loss=alternative,
                            conditioning_key=context,
                            max_variables=config["max_transport_variables"],
                            statistic=statistic,
                        )
                        noisy_records.append(
                            {
                                "statistic": statistic,
                                "query_id": q,
                                "radius": radius,
                                "regret": result["worst_report_regret"],
                                "replicate_count": replicates,
                            }
                        )
                direct_replicates = max(
                    1, replicates * config["local_reported_scalars"] // n
                )
                direct_range = uniform_goal_sample_range(
                    synthesis, reference, threshold
                )
                direct_radius = (direct_range[1] - direct_range[0]) * math.sqrt(
                    math.log(2 * (2 * qcount + 1) / config["failure_probability"])
                    / (2 * direct_replicates)
                )
                direct_matched = worst_report_decision_regret(
                    problem,
                    (0,),
                    np.array([direct_radius]),
                    alternative_loss=alternative,
                    conditioning_key=context,
                    max_variables=config["max_transport_variables"],
                    statistic="direct_goal",
                )
                noisy_records.append(
                    {
                        "statistic": "direct_goal_matched_scalar_budget",
                        "query_id": 0,
                        "radius": direct_radius,
                        "regret": direct_matched["worst_report_regret"],
                        "replicate_count": direct_replicates,
                    }
                )
                noisy_innovations = [
                    r for r in noisy_records if r["statistic"] == "innovation"
                ]
                best_noisy = min(noisy_innovations, key=lambda r: r["regret"])[
                    "query_id"
                ]
                law_records = []
                for kind in config["source_laws"]:
                    indices, masses = quantile_source_law(p, kind, seed)
                    fields = reference + code.numpy()[indices] @ synthesis.T
                    if fields.min() < 0:
                        raise ArithmeticError(
                            "Finite-law clean fields violated their declared physical floor."
                        )
                    values = evaluate_haar_innovations(
                        problem, indices, conditioning_key=context
                    )
                    expected = (
                        problem.offset
                        + np.maximum(threshold, fields).mean(axis=1) @ masses
                    )
                    if not math.isclose(
                        expected,
                        problem.root_risk + float(masses @ values.sum(1)),
                        abs_tol=1e-11,
                    ):
                        raise ArithmeticError(
                            "Source risk and innovation decomposition disagree."
                        )
                    draw = rng.choice(len(indices), size=replicates, p=masses)
                    means = _sample_block_means(
                        fields[draw], problem.node_ids, len(problem.lengths)
                    )
                    report = report_from_matched_blocks(
                        problem,
                        tuple(range(qcount)),
                        means,
                        measurement_error=np.zeros((qcount, 2)),
                        coefficient_error_bound=0,
                        failure_probability=config["failure_probability"],
                        conditioning_key=context,
                    )
                    true_moments = masses @ values
                    coverage = all(
                        lo <= true_moments[q] + 1e-12 and true_moments[q] - 1e-12 <= hi
                        for q, (lo, hi) in report["intervals"].items()
                    )
                    outcomes = {}
                    for label, q in [
                        ("regret_selected", best_noisy),
                        ("energy_selected", energy_query),
                        ("first_node", 0),
                    ]:
                        try:
                            conditioned = condition_haar_risk(
                                problem,
                                bind_haar_moment_intervals(
                                    problem, {q: report["intervals"][q]}
                                ),
                                conditioning_key=context,
                            )
                            interval = [conditioned["lower"], conditioned["upper"]]
                            outcomes[label] = {
                                "query_id": q,
                                "interval": interval,
                                **evaluate_binary_decision(
                                    interval, alternative, expected
                                ),
                                "covers_source_risk": bool(
                                    interval[0] - 1e-10
                                    <= expected
                                    <= interval[1] + 1e-10
                                ),
                            }
                        except ArithmeticError as error:
                            outcomes[label] = {
                                "query_id": q,
                                "status": "infeasible_report",
                                "message": str(error),
                            }
                    law_records.append(
                        {
                            "law": kind,
                            "support_count": len(masses),
                            "physical_min": float(fields.min()),
                            "true_risk": float(expected),
                            "simultaneous_moment_coverage": coverage,
                            "outcomes": outcomes,
                        }
                    )
                sampling_records.append(
                    {
                        "replicate_count": replicates,
                        "queries": noisy_records,
                        "source_laws": law_records,
                        "summary_channel_local_scalar_cost": config[
                            "local_reported_scalars"
                        ]
                        * replicates,
                        "summary_channel_direct_scalar_cost": n * replicates,
                        "full_trace_scalar_cost_all_methods": n * replicates,
                    }
                )
            record = {
                "time_count": n,
                "seed": seed,
                "threshold": threshold,
                "conditioning_key": context,
                "initial_interval": [initial["lower"], initial["upper"]],
                "alternative_loss": alternative,
                "initial_regret": _risk_regret(
                    [initial["lower"], initial["upper"]], alternative
                ),
                "variables": problem.variable_count,
                "paired_variables": 2 * problem.variable_count + 1,
                "exact_queries": exact_records,
                "best_exact_query": best_exact,
                "energy_query": energy_query,
                "uninformed_mean_regret": float(innovation_scores.mean()),
                "energy_regret": float(innovation_scores[energy_query]),
                "tail_certificates": [
                    haar_energy_tail_bound(problem, d)
                    for d in range(len(problem.lengths) + 1)
                ],
                "sampling": sampling_records,
                "wall_seconds": time.perf_counter() - start,
            }
            records.append(record)
            print(
                json.dumps(
                    {
                        "time_count": n,
                        "seed": seed,
                        "initial_regret": record["initial_regret"],
                        "best_exact_regret": float(innovation_scores.min()),
                        "energy_regret": record["energy_regret"],
                        "wall_seconds": record["wall_seconds"],
                    }
                ),
                flush=True,
            )
    result = {
        "status": "completed_finite_law_evaluation",
        "scope": "generated_matched_measurement_channels_not_biological_evidence",
        "date": "2026-09-11",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": platform.node(),
        "versions": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
        },
        "config": config,
        "records": records,
        "cost_scope": "reported_scalar_counts_under_declared_summary_or_full_trace_access_not_measured_assay_cost",
        "selection_scope": "single_precommitted_query_from_explicit_menu_not_adaptive_policy",
        "access_family": evaluate_access_family(config["access_family"]),
    }
    validate_evaluation(result)
    return result


def validate_evaluation(result):
    config = result["config"]
    access = result["access_family"]
    if len(access["records"]) != len(config["access_family"]["time_counts"]):
        raise ValueError("Incomplete access-family horizon coverage.")
    if access["pairing_necessity"] or access["biological_sample_advantage"]:
        raise ValueError(
            "The constructed family does not support a pairing/sample advantage."
        )
    if any(not all(row["checks"].values()) for row in access["records"]):
        raise ValueError("Failed access-family identities.")
    if len(result["records"]) != len(config["time_counts"]) * len(config["seeds"]):
        raise ValueError("Missing finite-law configurations.")
    for record in result["records"]:
        if record["paired_variables"] != 2 * record["variables"] + 1:
            raise ValueError("Paired LP size mismatch.")
        expected_queries = record["time_count"] - 1
        if len(record["exact_queries"]) != 2 * expected_queries + 1:
            raise ValueError("Incomplete matched query menu.")
        if any(
            not math.isfinite(q["regret"]) or q["regret"] < 0
            for q in record["exact_queries"]
        ):
            raise ValueError("Nonfinite/negative query regret.")
        direct = next(
            q for q in record["exact_queries"] if q["statistic"] == "direct_goal"
        )
        if direct["regret"] > 1e-8:
            raise ValueError("Exact direct goal must identify the comparison.")
        for sampling in record["sampling"]:
            if len(sampling["source_laws"]) != len(config["source_laws"]):
                raise ValueError("Missing source-law records.")
            for law in sampling["source_laws"]:
                for outcome in law["outcomes"].values():
                    if outcome.get("status") == "infeasible_report":
                        continue
                    expected = evaluate_binary_decision(
                        outcome["interval"],
                        record["alternative_loss"],
                        law["true_risk"],
                    )
                    if outcome["decision"] != expected["decision"]:
                        raise ValueError(
                            "Reported choice differs from deterministic minimax decision."
                        )
                    for metric in (
                        "conditional_minimax_regret",
                        "realized_decision_regret",
                    ):
                        if (
                            not math.isfinite(outcome[metric])
                            or abs(outcome[metric] - expected[metric]) > 1e-12
                        ):
                            raise ValueError("Incorrect decision-regret semantics.")
    return True


def evaluate_posterior_bridge(config):
    """Shared-context counterexample and bounded posterior scaling checks."""
    code = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)

    def cell(n, probabilities, offset, context, first_leaf=False):
        codec = GraphSpectralTemporalWaveletTokenizer(4, 2, 16, 1, False)
        metadata = codec.pack_temporal_wavelets(code.new_zeros((1, n, 1, 1)))[1]
        weights = code.new_full((n,), 1 / n)
        if first_leaf:
            weights[:] = 0
            weights[0] = math.sqrt(2)
        return compile_haar_dependence_queries(
            code.new_tensor(probabilities),
            code,
            metadata,
            code.new_zeros(n),
            weights,
            code.new_tensor(offset),
            lower_bound=0,
            conditioning_key=context,
            max_variables=config["max_variables"],
        )

    conditional = tuple(
        (
            cell(2, [[0.5, 0, 0.5]] * 2, baseline - 0.6, f"context{z}_candidate", True),
            cell(2, [[0, 1, 0]] * 2, baseline, f"context{z}_comparator", True),
        )
        for z, baseline in enumerate((1.0, 11.0))
    )
    problem = compile_posterior_haar(
        conditional,
        [0.5, 0.5],
        posterior_tv_radius=0.25,
        posterior_key="shared_context_demonstration",
        context_ids=("low_burden", "high_burden"),
        action_ids=("candidate", "comparator"),
        max_variables=config["max_variables"],
    )
    query = PosteriorMomentQuery("candidate", 0)
    report = bind_posterior_report(problem, (query,), [[0.25, 0.35]])
    before = posterior_contrast_bounds(problem, ("candidate", "comparator"))
    after = posterior_contrast_bounds(
        problem, ("candidate", "comparator"), report=report
    )
    cost_error = posterior_contrast_bounds(
        problem,
        ("candidate", "comparator"),
        report=report,
        conditional_loss_errors=np.full((2, 2), 0.02),
    )
    query_error_report = bind_posterior_report(
        problem,
        (query,),
        [[0.25, 0.35]],
        conditional_query_errors=[[0.02, 0.02]],
    )
    query_error = posterior_contrast_bounds(
        problem, ("candidate", "comparator"), report=query_error_report
    )
    menu = select_posterior_query(
        problem,
        ("candidate", "comparator"),
        ((query,),),
        (np.array([0.05]),),
        np.array([0.01]),
        max_variables=config["max_variables"],
    )
    error_menu = select_posterior_query(
        problem,
        ("candidate", "comparator"),
        ((query,),),
        (np.array([0.02]),),
        np.array([0.35]),
        max_variables=config["max_variables"],
        conditional_loss_errors=np.array([[0.02, 0.03], [0.10, 0.20]]),
        conditional_query_errors=(np.array([[0.04, 0.12]]),),
    )
    if error_menu["selected_index"] != 0:
        raise ArithmeticError(
            "Integrated error allowances should favor no query in the declared costly case."
        )
    expected = [(-0.6, 0.4), (0.15, 0.25), (0.11, 0.29), (0.13, 0.27)]
    for measured, target in zip(
        (before, after, cost_error, query_error), expected, strict=True
    ):
        if not np.allclose(
            [measured["lower"], measured["upper"]], target, atol=1e-8, rtol=0
        ):
            raise ArithmeticError(
                "Posterior counterexample disagrees with its analytic values."
            )
    if (
        menu["selected_index"] != 1
        or abs(menu["selected"]["worst_report_regret"] - 0.05) > 1e-8
    ):
        raise ArithmeticError(
            "Noisy pooled query failed its analytic regret comparison."
        )
    records = []
    for n in config["time_counts"]:
        for contexts in config["context_counts"]:
            for tv in config["posterior_tv_radii"]:
                start = time.perf_counter()
                rng = np.random.default_rng(config["seed"] + 10 * n + contexts)
                cells = []
                for z in range(contexts):
                    row = []
                    for a in range(2):
                        spread = rng.uniform(0.1, 0.45, n)
                        probabilities = np.stack(
                            [spread, 1 - 2 * spread, spread], axis=1
                        )
                        row.append(
                            cell(
                                n,
                                probabilities,
                                10 * z + 0.02 * a,
                                f"time{n}_state{z}_action{a}",
                            )
                        )
                    cells.append(tuple(row))
                mixture = compile_posterior_haar(
                    tuple(cells),
                    np.full(contexts, 1 / contexts),
                    posterior_tv_radius=tv,
                    posterior_key=f"time{n}_contexts{contexts}_tv{tv}",
                    context_ids=tuple(f"context{z}" for z in range(contexts)),
                    action_ids=("candidate", "comparator"),
                    max_variables=config["max_variables"],
                )
                bounds = posterior_contrast_bounds(mixture, ("candidate", "comparator"))
                no_query = posterior_worst_report_regret(
                    mixture,
                    ("candidate", "comparator"),
                    (),
                    np.empty(0),
                    max_variables=config["max_variables"],
                )
                measured = posterior_worst_report_regret(
                    mixture,
                    ("candidate", "comparator"),
                    (PosteriorMomentQuery("candidate", 0),),
                    np.array([config["report_radius"]]),
                    max_variables=config["max_variables"],
                )
                if (
                    measured["worst_report_regret"]
                    > no_query["worst_report_regret"] + 1e-8
                ):
                    raise ArithmeticError(
                        "A report must not worsen the best conditional decision."
                    )
                records.append(
                    {
                        "time_count": n,
                        "context_count": contexts,
                        "posterior_tv_radius": tv,
                        "single_variables": mixture.variable_count,
                        "paired_variables": 2 * mixture.variable_count + 1,
                        "contrast": bounds,
                        "no_query": no_query,
                        "pooled_innovation_query": measured,
                        "wall_seconds": time.perf_counter() - start,
                    }
                )
    return {
        "status": "verified_finite_posterior_bridge",
        "date": "2026-09-14",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": platform.node(),
        "config": config,
        "counterexample": {
            "before": before,
            "pooled_report": after,
            "cost_error_allowance": cost_error,
            "query_error_allowance": query_error,
            "query_menu": menu,
            "error_aware_query_menu": error_menu,
            "analytic_exact_report_independent_endpoint_contrast": [-4.8, 5.2],
            "analytic_exact_report_shared_contrast": [0.2, 0.2],
        },
        "records": records,
        "scope": "constructed_conditional_marginal_classes_not_a_learned_or_biological_posterior",
        "reference_risk_qualified": False,
    }


def _growing_depth_pairs(depth, theta):
    """Exact law of two child means at a uniformly chosen node of one level.

    This is the manuscript's bounded physical family. Equal-law right nodes
    share a path distribution; this oracle avoids enumerating entire fields.
    """
    threshold, displacement, baseline_drop, detail_scale = 0.25, 0.25, 1 / 16, 0.01
    a = np.array([0.0, displacement, 0.0, displacement])
    d = np.array([0.0, displacement, displacement, 0.0])
    component_mass = np.array([1 - theta, 1 - theta, theta, theta]) / 2
    if depth == 0:
        return (
            threshold + np.repeat(a, 2),
            threshold + np.tile([0.0, -baseline_drop], 4),
            np.repeat(component_mass / 2, 2),
        )
    if depth == 1:
        left_plus = threshold + a + d
        left_minus = threshold + a - d
        left_mass = component_mass / 2
    else:
        left_plus = threshold + np.concatenate([a + d, a - d])
        left_minus = left_plus.copy()
        left_mass = np.tile(component_mass / 4, 2)
    path_values, path_mass = np.array([0.0]), np.array([1.0])
    for level in range(depth - 1):
        shift = detail_scale * 2 ** (-level / 2)
        path_values = np.concatenate(
            [path_values - shift, path_values, path_values + shift]
        )
        path_mass = np.concatenate([path_mass / 4, path_mass / 2, path_mass / 4])
    node_mean = threshold + np.concatenate([path_values, path_values - baseline_drop])
    node_mass = np.tile(path_mass / 2, 2)
    shift = detail_scale * 2 ** (-(depth - 1) / 2)
    right_plus = np.concatenate([node_mean - shift, node_mean, node_mean + shift])
    right_minus = np.concatenate([node_mean + shift, node_mean, node_mean - shift])
    right_mass = np.concatenate([node_mass / 8, node_mass / 4, node_mass / 8])
    return (
        np.concatenate([left_plus, right_plus]),
        np.concatenate([left_minus, right_minus]),
        np.concatenate([left_mass, right_mass]),
    )


def evaluate_multiscale_sampling(config):
    """Check exact moments and noisy reports on a genuinely growing-depth law."""
    threshold, displacement, baseline_drop, detail_scale = 0.25, 0.25, 1 / 16, 0.01
    energy_constant = max(
        (displacement**2 + baseline_drop**2 + displacement * baseline_drop) / 8,
        (displacement**2 + detail_scale**2) / 2,
    )
    root_risk = (2 * displacement - baseline_drop) / 8
    records = []
    for horizon in config["time_counts"]:
        depth_count = int(math.log2(horizon))
        path_values, path_mass = np.array([0.0]), np.array([1.0])
        for level in range(depth_count - 1):
            shift = detail_scale * 2 ** (-level / 2)
            path_values = np.concatenate(
                [path_values - shift, path_values, path_values + shift]
            )
            path_mass = np.concatenate([path_mass / 4, path_mass / 2, path_mass / 4])
        right_risk = float(path_mass @ np.abs(path_values)) / 8
        for theta in config["mixture_weights"]:
            exact_risk = displacement / 4 + theta * displacement / 8 + right_risk
            for error_rms in config["mean_error_rms"]:
                start = time.perf_counter()
                design = compile_innovation_design(
                    horizon,
                    energy_constant,
                    0.5,
                    mean_error_rms=error_rms,
                    error_decay=config["error_decay"],
                )
                exact_mean = exact_second = clean_sum = 0.0
                active_levels = []
                level_sources = []
                for level, q in enumerate(design.level_probabilities):
                    plus, minus, masses = _growing_depth_pairs(level, theta)
                    if not np.isclose(masses.sum(), 1.0):
                        raise ArithmeticError(
                            "Physical channel mass is not normalized."
                        )
                    levels = np.full(len(masses), level, dtype=np.int64)
                    locations = InnovationLocations(
                        levels,
                        np.zeros(len(masses), dtype=np.int64),
                        np.full(len(masses), q / 2**level),
                    )
                    clean_y = innovation_importance_values(
                        design, locations, plus, minus, threshold=threshold
                    )
                    contribution = float(masses @ clean_y) * q
                    clean_sum += contribution
                    if contribution > 0:
                        active_levels.append(level)
                    error = error_rms * 2 ** (-config["error_decay"] * level)
                    for sign_plus, sign_minus in itertools.product([-1, 1], repeat=2):
                        noisy_y = innovation_importance_values(
                            design,
                            locations,
                            plus + sign_plus * error,
                            minus + sign_minus * error,
                            threshold=threshold,
                        )
                        exact_mean += q * float(masses @ noisy_y) / 4
                        exact_second += q * float(masses @ (noisy_y**2)) / 4
                    level_sources.append((plus, minus, masses))
                if abs(root_risk + clean_sum - exact_risk) > 1e-12:
                    raise ArithmeticError(
                        "Randomized mean disagrees with the leaf-risk oracle."
                    )
                bias = abs(root_risk + exact_mean - exact_risk)
                if bias > design.measurement_bias_bound + 1e-12:
                    raise ArithmeticError(
                        "Constructed noise exceeds the declared bias bound."
                    )
                if exact_second > design.second_moment_bound + 1e-12:
                    raise ArithmeticError(
                        "Exact second moment exceeds the theorem envelope."
                    )
                if len(active_levels) != depth_count:
                    raise ArithmeticError(
                        "The growing-depth family lost an active level."
                    )
                rng = np.random.default_rng(config["seed"] + horizon)
                locations = sample_innovation_locations(
                    design, config["replicates"], rng=rng
                )
                plus = np.empty(config["replicates"])
                minus = np.empty(config["replicates"])
                for level in range(depth_count):
                    mask = locations.levels == level
                    count = int(mask.sum())
                    node_indices = locations.node_indices[mask]
                    a = displacement * rng.integers(0, 2, size=count)
                    d = np.where(rng.random(count) < theta, displacement - a, a)
                    baseline = -baseline_drop * rng.integers(0, 2, size=count)
                    if level == 0:
                        lp, lm = threshold + a, threshold + baseline
                    else:
                        left = node_indices < 2 ** (level - 1)
                        if level == 1:
                            lp, lm = threshold + a + d, threshold + a - d
                        else:
                            signs = np.where(node_indices < 2 ** (level - 2), 1, -1)
                            lp = threshold + a + signs * d
                            lm = lp.copy()
                        right_mean = threshold + baseline
                        for ancestor in range(level - 1):
                            # Symmetric IID details have the same path law for
                            # either branch sign at the selected physical node.
                            right_mean = right_mean + detail_scale * 2 ** (
                                -ancestor / 2
                            ) * rng.choice([-1, 0, 1], size=count, p=[0.25, 0.5, 0.25])
                        detail = (
                            detail_scale
                            * 2 ** (-(level - 1) / 2)
                            * rng.choice([-1, 0, 1], size=count, p=[0.25, 0.5, 0.25])
                        )
                        lp = np.where(left, lp, right_mean + detail)
                        lm = np.where(left, lm, right_mean - detail)
                    error = error_rms * 2 ** (-config["error_decay"] * level)
                    plus[mask] = lp + error * rng.choice([-1, 1], size=count)
                    minus[mask] = lm + error * rng.choice([-1, 1], size=count)
                estimate = estimate_innovation_risk(
                    design,
                    locations,
                    plus,
                    minus,
                    threshold=threshold,
                    root_risk=root_risk,
                    alpha=config["failure_probability"],
                )
                records.append(
                    {
                        "time_count": horizon,
                        "active_depth": len(active_levels),
                        "mixture_weight": theta,
                        "mean_error_rms": error_rms,
                        "exact_risk": exact_risk,
                        "randomized_mean": root_risk + exact_mean,
                        "exact_measurement_bias": bias,
                        "exact_second_moment": exact_second,
                        "second_moment_bound": design.second_moment_bound,
                        "measurement_bias_bound": design.measurement_bias_bound,
                        "sampling_distribution_entries": len(
                            design.level_probabilities
                        ),
                        "report": estimate,
                        "wall_seconds": time.perf_counter() - start,
                    }
                )
    return {
        "status": "verified_growing_depth_moment_identities",
        "date": "2026-09-14",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": platform.node(),
        "config": config,
        "records": records,
        "scope": "exact_constructed_law_moments_and_IID_noise_not_empirical_calibration_or_assay_cost",
        "reference_risk_qualified": False,
    }


def evaluate_observation_experiments(config):
    """Matched readout information in the exact four-leaf marginal class."""
    code = torch.arange(-1, 9, dtype=torch.float64)
    marginals = code.new_zeros((4, 10))
    marginals[0, [7, 9]] = 0.5
    marginals[1, [0, 2]] = 0.5
    marginals[2, 3] = 1.0
    marginals[3, 1] = 1.0
    codec = GraphSpectralTemporalWaveletTokenizer(4, 2, 8, 1.0, False)
    metadata = codec.pack_temporal_wavelets(code.new_zeros((1, 4, 1, 1)))[1]
    problem = compile_haar_dependence_queries(
        marginals,
        code,
        metadata,
        code.new_full((4,), -2.0),
        code.new_full((4,), 0.25),
        code.new_zeros(()),
        lower_bound=0.0,
        conditioning_key="four_leaf_observation_information",
        max_variables=config["max_variables"],
    )
    root = problem.coupling.tables[0]
    cells = root.columns.stop - root.columns.start
    root_selector = coo_matrix(
        (
            np.ones(cells),
            (np.arange(cells), np.arange(root.columns.start, root.columns.stop)),
        ),
        shape=(cells, problem.variable_count),
    ).tocsr()

    def law(rho):
        mass = np.zeros(cells)
        for a, d, p in [
            (6, 1, (1 - rho) / 2),
            (8, -1, (1 - rho) / 2),
            (6, -1, rho / 2),
            (8, 1, rho / 2),
        ]:
            ai = np.flatnonzero(root.amplitudes == a).item()
            di = np.flatnonzero(root.details == d).item()
            mass[ai * len(root.details) + di] = p
        result = linprog(
            np.zeros(problem.variable_count),
            A_eq=vstack((problem.coupling.matrix, root_selector)),
            b_eq=np.r_[problem.coupling.rhs, mass],
            bounds=(0, None),
            method="highs",
        )
        if not result.success:
            raise ArithmeticError(result.message)
        return result.x

    root_pairs = list(itertools.product(root.amplitudes, root.details))
    # Declared functions of the physical readout; partitions precede samples.
    root_dependence = compile_haar_observation_channel(
        problem,
        0,
        labels=["central_mean" if a + d == 7 else "outer_mean" for a, d in root_pairs],
    )
    raw_mean = compile_haar_observation_channel(
        problem,
        0,
        labels=[
            "low" if a + d < 7 else "central" if a + d == 7 else "high"
            for a, d in root_pairs
        ],
    )
    root_innovation = compile_haar_observation_channel(
        problem,
        0,
        labels=[
            str(max(a + d - 4, 0) + max(a - d - 4, 0) - 2 * max(a - 4, 0))
            for a, d in root_pairs
        ],
    )
    left_index = problem.node_ids.index((1, 0))
    left = problem.coupling.tables[left_index]
    left_labels = []
    for a, d in itertools.product(left.amplitudes, left.details):
        twice_mean = int(round(math.sqrt(2) * a))
        if abs(math.sqrt(2) * a - twice_mean) > 1e-12:
            raise ArithmeticError(
                "Four-leaf readout coordinate is not on its exact integer support."
            )
        left_labels.append("crossing" if twice_mean == 5 and d == 2 else "zero")
    left_innovation = compile_haar_observation_channel(
        problem, left_index, labels=left_labels
    )
    channels = {
        "root_cell": compile_haar_observation_channel(problem, 0),
        "root_dependence": root_dependence,
        "raw_half_mean": raw_mean,
        "root_innovation": root_innovation,
        "left_innovation": left_innovation,
    }
    delta = (2 * math.sqrt(2) - 1) / 16
    boundary = 1.5 + delta / 2
    alternative = law(0.5)
    records = []
    for rho in config["mixture_weights"]:
        source = law(rho)
        exact_risk = 1.5 + rho * delta
        if abs(problem.coupling.lower_cost @ source - exact_risk) > 1e-11:
            raise ArithmeticError(
                "Compressed source disagrees with analytical field loss."
            )
        full_information = categorical_kl([rho, 1 - rho], [0.5, 0.5])
        scalar_information = categorical_kl([rho / 2, 1 - rho / 2], [0.25, 0.75])
        for retention in config["retention_probabilities"]:
            rng = np.random.default_rng(
                config["seed"] + int(1000 * rho) + int(100 * retention)
            )
            source_pairs = np.array([[6, 1], [8, -1], [6, -1], [8, 1]])
            source_draws = rng.choice(
                4,
                size=max(config["sample_counts"]),
                p=[(1 - rho) / 2, (1 - rho) / 2, rho / 2, rho / 2],
            )
            erased = rng.random(len(source_draws)) >= retention
            record = {
                "mixture_weight": rho,
                "retention_probability": retention,
                "exact_risk": exact_risk,
                "decision_threshold": boundary,
                "information_to_boundary": {},
                "confidence_reports": {},
            }
            for name, channel in channels.items():
                table = problem.coupling.tables[channel.node_index]
                kernel = channel.operator[:, table.columns]
                noisy = compile_haar_observation_channel(
                    problem,
                    channel.node_index,
                    kernel=vstack(
                        (
                            retention * kernel,
                            csr_matrix(np.full((1, kernel.shape[1]), 1 - retention)),
                        )
                    ),
                    outcome_ids=(*channel.outcome_ids, "erased"),
                )
                p = observation_probabilities(problem, noisy, source)
                q = observation_probabilities(problem, noisy, alternative)
                information = categorical_kl(p, q)
                expected = retention * (
                    0.0
                    if name == "root_innovation"
                    else scalar_information
                    if name == "left_innovation"
                    else full_information
                )
                if abs(information - expected) > 1e-11:
                    raise ArithmeticError(
                        "Readout information disagrees with analytical KL."
                    )
                record["information_to_boundary"][name] = information
                if name in {"root_dependence", "raw_half_mean", "left_innovation"}:
                    atom_outcomes = []
                    for a, d in source_pairs:
                        amplitude, detail = (
                            (a, d)
                            if channel.node_index == 0
                            else ((a + d) / math.sqrt(2), 2)
                        )
                        ai = np.flatnonzero(
                            np.isclose(table.amplitudes, amplitude, rtol=0, atol=1e-12)
                        ).item()
                        di = np.flatnonzero(table.details == detail).item()
                        column = table.columns.start + ai * len(table.details) + di
                        probabilities = channel.operator[:, column].toarray().ravel()
                        if (
                            np.count_nonzero(probabilities) != 1
                            or probabilities.max() != 1
                        ):
                            raise ArithmeticError(
                                "Matched readouts require the declared deterministic pre-erasure channel."
                            )
                        atom_outcomes.append(int(np.argmax(probabilities)))
                    draws = np.asarray(atom_outcomes)[source_draws]
                    draws = np.where(erased, len(p) - 1, draws)
                    reports = []
                    for count in config["sample_counts"]:
                        observed = np.bincount(draws[:count], minlength=len(p))
                        bound = haar_observation_risk_bounds(
                            problem,
                            (noisy,),
                            (bind_haar_observation_counts(problem, noisy, observed),),
                            alpha=config["failure_probability"],
                            decision_threshold=boundary,
                        )
                        bound["contains_source_risk"] = (
                            bound["lower"] <= exact_risk <= bound["upper"]
                        )
                        reports.append(bound)
                    record["confidence_reports"][name] = reports
            direct = [
                retention * (1 - rho) / 2,
                retention * rho / 2,
                retention / 2,
                1 - retention,
            ]
            direct_boundary = [
                retention / 4,
                retention / 4,
                retention / 2,
                1 - retention,
            ]
            record["information_to_boundary"]["direct_goal"] = categorical_kl(
                direct, direct_boundary
            )
            if (
                abs(
                    record["information_to_boundary"]["direct_goal"]
                    - retention * full_information / 2
                )
                > 1e-11
            ):
                raise ArithmeticError(
                    "Direct-goal information disagrees with independent field law."
                )
            records.append(record)
    return {
        "status": "verified_observation_information_experiments",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": platform.node(),
        "config": config,
        "records": records,
        "table_variables": problem.variable_count,
        "scope": "exact_four_leaf_class_readout_information_and_conservative_anytime_confidence",
        "measurement_matching": "same_field_replicas_and_same_erasure_events_for_every_readout",
        "claims_excluded": [
            "asymptotically_optimal_policy",
            "empirical_coverage_estimation",
            "biological_calibration",
            "measured_assay_cost",
        ],
        "reference_risk_qualified": False,
    }


def evaluate_joint_observation(config):
    """Dense-law validation of same-trajectory readouts with fixed local tables.

    The four-leaf witness uses a physical reference and threshold both equal to
    one; its goal is the first-leaf exceedance plus the configured loss offset.
    Full coefficient laws provide an independent 81-variable comparison.
    """
    code = torch.tensor([-1.0, 0.0, 1.0], dtype=torch.float64)
    marginals = code.new_tensor(
        [[0, 0.5, 0.5], [0, 0.5, 0.5], [0.5, 0, 0.5], [0, 1, 0]]
    )
    codec = GraphSpectralTemporalWaveletTokenizer(4, 2, 8, 1.0, False)
    metadata = codec.pack_temporal_wavelets(code.new_zeros((1, 4, 1, 1)))[1]
    synthesis = np.array(
        [
            [0.5, 0.5, 1 / math.sqrt(2), 0],
            [0.5, 0.5, -1 / math.sqrt(2), 0],
            [0.5, -0.5, 0, 1 / math.sqrt(2)],
            [0.5, -0.5, 0, -1 / math.sqrt(2)],
        ]
    )
    codec_error = float(
        np.max(np.abs(temporal_synthesis_matrix(codec, 4, code).numpy() - synthesis))
    )
    if codec_error > 2e-14:
        raise ArithmeticError("Codec differs from independent orthonormal synthesis.")
    positive_loss = 0.5 + 1 / math.sqrt(2)
    laws = (
        np.array([[0, 1, -1, 0], [1, 0, 1, 0]], dtype=float),
        np.array([[0, 1, 1, 0], [1, 0, -1, 0]], dtype=float),
    )
    full_indices = np.array(list(itertools.product(range(3), repeat=4)))
    full_states = code.numpy()[full_indices]
    full_fields = full_states @ synthesis.T
    full_joint = ((full_states[:, 0] == 0) & (full_states[:, 2] == -1)).astype(float)
    joint_memory = ("A=0", "detail=-1")
    reports = [("unreported", None)]
    reports.extend(("exact", (value, value)) for value in config["joint_probabilities"])
    reports.extend(("interval", tuple(value)) for value in config["joint_intervals"])
    records, witnesses = [], []
    for offset in config["loss_offsets"]:
        if not math.isfinite(offset):
            raise ValueError("Loss offsets must be finite.")
        weights = code.new_tensor([1, 0, 0, 0])
        problem = compile_haar_dependence_queries(
            marginals,
            code,
            metadata,
            code.new_zeros(4),
            weights,
            code.new_tensor(offset),
            lower_bound=0,
            conditioning_key="joint_readout_four_leaf_exceedance",
            max_variables=config["max_variables"],
        )
        starts = np.cumsum((0, *problem.sizes))

        def project(states):
            fields = states @ synthesis.T
            result = np.zeros(problem.variable_count)
            for table in problem.coupling.tables:
                width = 4 // 2**table.depth
                start = table.position * width
                amplitudes = fields[:, start : start + width].mean(axis=1) * math.sqrt(
                    width
                )
                details = states[:, starts[table.depth + 1] + table.position]
                for amplitude, detail in zip(amplitudes, details, strict=True):
                    matches = np.flatnonzero(
                        np.isclose(table.amplitudes, amplitude, rtol=0, atol=2e-14)
                    )
                    if len(matches) != 1:
                        raise ArithmeticError(
                            "Independently synthesized amplitude has no unique exact-grid cell."
                        )
                    detail_index = np.flatnonzero(table.details == detail).item()
                    result[
                        table.columns.start
                        + matches.item() * len(table.details)
                        + detail_index
                    ] += 0.5
            return result

        prior, other = map(project, laws)
        same_table_error = float(np.max(np.abs(prior - other)))
        prior_residual = float(
            np.max(np.abs(problem.coupling.matrix @ prior - problem.coupling.rhs))
        )
        if max(same_table_error, prior_residual) > 2e-14:
            raise ArithmeticError("Witness laws must share every original local table.")
        root = problem.coupling.tables[0]
        left_index = problem.node_ids.index((1, 0))
        left = problem.coupling.tables[left_index]
        channels = (
            compile_haar_observation_channel(
                problem,
                0,
                labels=[f"A={int(a)}" for a in root.amplitudes for _ in root.details],
            ),
            compile_haar_observation_channel(
                problem,
                left_index,
                labels=[
                    f"detail={int(d)}" for _ in left.amplitudes for d in left.details
                ],
            ),
        )
        lift = compile_joint_observation(
            problem, channels, max_variables=config["max_variables"]
        )
        source_laws = []
        for name, states in zip(
            ("low_conditional_loss", "high_conditional_loss"), laws, strict=True
        ):
            fields = states @ synthesis.T
            losses = np.maximum(fields[:, 0], 0) + offset
            event = states[:, 0] == 0
            joint = event & (states[:, 2] == -1)
            source_laws.append(
                {
                    "name": name,
                    "joint_probability": float(joint.mean()),
                    "unconditional_loss": float(losses.mean()),
                    "event_probability": float(event.mean()),
                    "event_weighted_loss": float((losses * event).mean()),
                    "conditional_loss": float(losses[event].mean()),
                    "conditional_decision": "candidate"
                    if losses[event].mean() < positive_loss / 2 + offset
                    else "comparator",
                    "minimum_physical_field": float((1 + fields).min()),
                }
            )
        witnesses.append(
            {
                "loss_offset": offset,
                "same_table_max_error": same_table_error,
                "prior_constraint_residual": prior_residual,
                "conditional_comparator": positive_loss / 2 + offset,
                "source_laws": source_laws,
            }
        )
        for fixed in (False, True):
            equalities, rhs = [], []
            for coordinate in range(4):
                for value in range(3):
                    equalities.append(
                        (full_indices[:, coordinate] == value).astype(float)
                    )
                    rhs.append(marginals[coordinate, value].item())
            if fixed:
                # Root pairs plus L/R singletons specify the original tables
                # in this witness while leaving the cross-node copula free.
                for a, d in itertools.product(code.numpy(), repeat=2):
                    equalities.append(
                        ((full_states[:, 0] == a) & (full_states[:, 1] == d)).astype(
                            float
                        )
                    )
                    rhs.append(0.5 if (a, d) in ((0, 1), (1, 0)) else 0)
            for event_name in ("all", "A0", "joint"):
                memories = (
                    None
                    if event_name == "all"
                    else tuple(
                        memory
                        for memory in lift.memory_ids
                        if memory[0] == "A=0"
                        and (event_name == "A0" or memory[1] == "detail=-1")
                    )
                )
                mask = (
                    np.ones(len(full_states))
                    if event_name == "all"
                    else (
                        full_joint
                        if event_name == "joint"
                        else (full_states[:, 0] == 0).astype(float)
                    )
                )
                for report_type, interval in reports:
                    actual = joint_observation_bounds(
                        lift,
                        memories,
                        prior_table_law=prior if fixed else None,
                        joint_probability_intervals=None
                        if interval is None
                        else {joint_memory: interval},
                    )
                    dense = []
                    for cost in (
                        mask,
                        (np.maximum(full_fields[:, 0], 0) + offset) * mask,
                    ):
                        for sign in (1, -1):
                            solved = linprog(
                                sign * cost,
                                A_eq=np.asarray(equalities),
                                b_eq=np.asarray(rhs),
                                A_ub=None
                                if interval is None
                                else np.stack((full_joint, -full_joint)),
                                b_ub=None
                                if interval is None
                                else [interval[1], -interval[0]],
                                bounds=(0, None),
                                method="highs",
                            )
                            if not solved.success:
                                raise ArithmeticError(
                                    "Dense joint-law oracle: " + solved.message
                                )
                            dense.append(float(sign * solved.fun))
                    values = [
                        actual[key]
                        for key in (
                            "probability_lower",
                            "probability_upper",
                            "event_loss_lower",
                            "event_loss_upper",
                        )
                    ]
                    oracle_error = float(np.max(np.abs(np.asarray(values) - dense)))
                    analytic = None
                    if fixed:
                        lo, hi = (0.0, 0.5) if interval is None else interval
                        if event_name == "all":
                            analytic = [
                                1,
                                1,
                                positive_loss / 2 + offset,
                                positive_loss / 2 + offset,
                            ]
                        elif event_name == "A0":
                            analytic = [
                                0.5,
                                0.5,
                                positive_loss * (0.5 - hi) + offset / 2,
                                positive_loss * (0.5 - lo) + offset / 2,
                            ]
                        else:
                            analytic = [lo, hi, *sorted((offset * lo, offset * hi))]
                    analytic_error = (
                        None
                        if analytic is None
                        else float(np.max(np.abs(np.asarray(values) - analytic)))
                    )
                    if oracle_error > 2e-9 or (
                        analytic_error is not None and analytic_error > 2e-9
                    ):
                        raise ArithmeticError(
                            "Memory lift disagrees with independent full-law or analytic bounds."
                        )
                    records.append(
                        {
                            "loss_offset": offset,
                            "aggregate_tables": "fixed" if fixed else "free",
                            "event": event_name,
                            "report_type": report_type,
                            "joint_probability_interval": interval,
                            "lift": actual,
                            "dense_bounds": dense,
                            "analytic_bounds": analytic,
                            "lift_dense_max_error": oracle_error,
                            "lift_analytic_max_error": analytic_error,
                        }
                    )
    return {
        "status": "verified_same_trajectory_joint_readout_bounds",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": platform.node(),
        "config": config,
        "time_count": 4,
        "coefficient_alphabet": code.tolist(),
        "dense_coefficient_states": len(full_states),
        "base_table_variables": problem.variable_count,
        "memory_count": lift.memory_count,
        "lift_variables": lift.variable_count,
        "positive_loss": positive_loss,
        "physical_reference": 1.0,
        "physical_threshold": 1.0,
        "codec_max_error": codec_error,
        "witnesses": witnesses,
        "records": records,
        "max_lift_dense_error": max(
            record["lift_dense_max_error"] for record in records
        ),
        "max_primal_residual": max(
            record["lift"]["primal_residual"] for record in records
        ),
        "max_numerical_duality_gap": max(
            record["lift"]["numerical_duality_gap"] for record in records
        ),
        "scope": "constructed_four_leaf_exact_grid_deterministic_same_trajectory_readout_memory",
        "report_interpretation": "externally_supplied_probability_intervals_not_estimated_confidence_intervals",
        "reference_risk_qualified": False,
        "calibration_established": False,
        "claims_excluded": [
            "biological_validation",
            "empirical_coverage",
            "optimal_adaptive_policy",
            "large_horizon_scaling",
        ],
    }


def _compression_fields(coefficients, protected, epsilon, rounding):
    """Full-history synthesis independent of the local-state compiler."""
    count, n = coefficients.shape
    levels = n.bit_length() - 1
    exact = {(0, 0)}
    for depth, position in protected:
        while depth:
            exact.add((depth, position))
            depth, position = depth - 1, position // 2
    true = {(0, 0): coefficients[:, 0]}
    proxy = {(0, 0): coefficients[:, 0]}
    gamma = 2**0.75
    total = sum(gamma**depth for depth in range(1, levels))
    for depth in range(levels):
        for position in range(2**depth):
            parent = (depth, position)
            detail = coefficients[:, 2**depth + position]
            for branch, sign in enumerate((1, -1)):
                child = depth + 1, 2 * position + branch
                true[child] = (true[parent] + sign * detail) / math.sqrt(2)
                values = (proxy[parent] + sign * detail) / math.sqrt(2)
                if child[0] < levels and child not in exact:
                    if rounding == "amplitude":
                        grid = (
                            2
                            * epsilon
                            * (math.sqrt(2) - 1)
                            / (1 - 2 ** (-(levels - 1) / 2))
                        )
                        values = np.floor(values / grid + 0.5) * grid
                    else:
                        grid = 2 * epsilon * gamma ** child[0] / total
                        size = n // 2 ** child[0]
                        values = (
                            np.floor(values / math.sqrt(size) / grid + 0.5)
                            * grid
                            * math.sqrt(size)
                        )
                proxy[child] = values
    fine_field = np.column_stack([true[(levels, i)] for i in range(n)])
    proxy_field = np.column_stack([proxy[(levels, i)] for i in range(n)])
    if fine_field.shape != (count, n):
        raise ArithmeticError("Invalid independent Haar synthesis shape.")
    return true, proxy, fine_field, proxy_field


def _compression_dense_extrema(cost, matrix, rhs):
    results = []
    for sign in (1, -1):
        solved = linprog(
            sign * cost, A_eq=matrix, b_eq=rhs, bounds=(0, None), method="highs"
        )
        if not solved.success:
            raise ArithmeticError("Dense compression oracle: " + solved.message)
        results.append(float(sign * solved.fun))
    return results


def evaluate_query_compression(config):
    """Compare exact queries with two independently budgeted rounding rules.

    Dense oracles optimize all original coefficient laws. Scaling records count
    actual reachable table entries and construction time; memory size there is
    algebraic, and large-horizon LP solve time is not measured.
    """
    budget = config["max_variables"]
    code = np.asarray(config["oracle_codebook"])
    records, scaling, witnesses = [], [], []
    for n in config["oracle_time_counts"]:
        levels, k = n.bit_length() - 1, len(code)
        query_nodes = ((0, 0), (levels - 1, 0))
        indices = np.array(list(itertools.product(range(k), repeat=n)))
        coefficients = code[indices]
        matrix = csr_matrix(
            np.array(
                [
                    (indices[:, coordinate] == value).astype(float)
                    for coordinate in range(n)
                    for value in range(k)
                ]
            )
        )
        probabilities = np.full((n, k), 1 / k)
        rhs = probabilities.ravel()
        reference = np.resize(config["reference_pattern"], n)
        weights = np.full(n, 1 / n)
        all_nodes = tuple((d, j) for d in range(levels) for j in range(2**d))
        for epsilon in config["oracle_epsilons"]:
            for mode in ("exact", "amplitude", "physical_mean"):
                rounding = "amplitude" if mode == "exact" else mode
                protected = all_nodes if mode == "exact" else query_nodes
                started = time.perf_counter()
                problem = compile_query_preserving_haar(
                    probabilities,
                    code,
                    reference,
                    weights,
                    lower_bound=config["lower_bound"],
                    offset=config["loss_offset"],
                    epsilon=epsilon,
                    protected_nodes=protected,
                    rounding=rounding,
                    max_variables=budget,
                )
                channels = (
                    compile_query_readout(problem, query_nodes[0], threshold=0),
                    compile_query_readout(
                        problem, query_nodes[1], detail_coefficient=0.25, threshold=0.13
                    ),
                )
                memory = compile_query_memory(
                    problem, channels, max_variables=config["max_memory_variables"]
                )
                build_seconds = time.perf_counter() - started
                true, proxy, fine_field, proxy_field = _compression_fields(
                    coefficients, protected, epsilon, rounding
                )
                labels = np.column_stack(
                    (
                        coefficients[:, 0] >= 0,
                        true[query_nodes[1]] + 0.25 * coefficients[:, 2 ** (levels - 1)]
                        >= 0.13,
                    )
                )
                fine_loss = (
                    np.maximum(config["lower_bound"], fine_field + reference) @ weights
                    + config["loss_offset"]
                )
                proxy_loss = (
                    np.maximum(config["lower_bound"], proxy_field + reference) @ weights
                    + config["loss_offset"]
                )
                field_error = float(np.max(abs(fine_field - proxy_field)))
                loss_error = float(np.max(abs(fine_loss - proxy_loss)))
                radius = float(abs(weights) @ problem.leaf_error_radii)
                if (
                    np.max(abs(fine_field - proxy_field) - problem.leaf_error_radii)
                    > 1e-11
                ):
                    raise ArithmeticError(
                        "Independent field error exceeds the analytic envelope."
                    )
                events = [("all", None)] + [
                    ("".join(map(str, r)), (r,)) for r in memory.memory_ids
                ]
                for name, selected in events:
                    mask = (
                        np.ones(len(indices))
                        if selected is None
                        else np.all(labels == selected[0], axis=1).astype(float)
                    )
                    started = time.perf_counter()
                    observed = query_memory_bounds(memory, selected)
                    local_seconds = time.perf_counter() - started
                    started = time.perf_counter()
                    dense = [
                        *_compression_dense_extrema(mask, matrix, rhs),
                        *_compression_dense_extrema(mask * proxy_loss, matrix, rhs),
                    ]
                    fine_bounds = _compression_dense_extrema(
                        mask * fine_loss, matrix, rhs
                    )
                    outer = [
                        _compression_dense_extrema(
                            mask * (proxy_loss - radius), matrix, rhs
                        )[0],
                        _compression_dense_extrema(
                            mask * (proxy_loss + radius), matrix, rhs
                        )[1],
                    ]
                    dense_seconds = time.perf_counter() - started
                    calculated = [
                        observed[key]
                        for key in (
                            "probability_lower",
                            "probability_upper",
                            "surrogate_event_loss_lower",
                            "surrogate_event_loss_upper",
                        )
                    ]
                    mismatch = float(np.max(abs(np.asarray(calculated) - dense)))
                    envelope_error = float(
                        np.max(
                            abs(
                                np.asarray(outer)
                                - [
                                    observed["true_event_loss_lower"],
                                    observed["true_event_loss_upper"],
                                ]
                            )
                        )
                    )
                    contains = (
                        observed["true_event_loss_lower"] <= fine_bounds[0] + 3e-9
                        and observed["true_event_loss_upper"] >= fine_bounds[1] - 3e-9
                    )
                    if max(mismatch, envelope_error) > 3e-9 or not contains:
                        raise ArithmeticError(
                            "Query compression disagrees with the independent law oracle."
                        )
                    records.append(
                        {
                            "time_count": n,
                            "alphabet_size": k,
                            "epsilon": epsilon,
                            "mode": mode,
                            "event": name,
                            "base_variables": problem.variable_count,
                            "memory_variables": memory.variable_count,
                            "dense_variables": len(indices),
                            "build_seconds": build_seconds,
                            "local_six_solves_seconds": local_seconds,
                            "dense_ten_solves_seconds": dense_seconds,
                            "bounds": observed,
                            "dense_probability_surrogate_bounds": dense,
                            "dense_true_loss_bounds": fine_bounds,
                            "dense_envelope_bounds": outer,
                            "oracle_max_error": mismatch,
                            "envelope_max_error": envelope_error,
                            "contains_true_extrema": contains,
                            "max_field_error": field_error,
                            "max_loss_error": loss_error,
                        }
                    )
                if mode != "exact":
                    # Deliberately round the queried path as an expected-failure control.
                    _, unprotected, _, _ = _compression_fields(
                        coefficients, ((0, 0),), epsilon, rounding
                    )
                    discrepancy = abs(
                        true[query_nodes[1]] - unprotected[query_nodes[1]]
                    )
                    index = int(np.argmax(discrepancy))
                    if discrepancy[index] > 1e-12:
                        a = float(true[query_nodes[1]][index])
                        b = float(unprotected[query_nodes[1]][index])
                        threshold = (a + b) / 2
                        witnesses.append(
                            {
                                "time_count": n,
                                "epsilon": epsilon,
                                "mode": mode,
                                "coefficient_assignment": coefficients[index].tolist(),
                                "fine_amplitude": a,
                                "rounded_amplitude": b,
                                "threshold": threshold,
                                "deterministic_readout_tv": 1,
                                "scope": "pointwise_discontinuity_witness_not_a_fixed_threshold_population_rate",
                            }
                        )
    for n in config["scaling_time_counts"]:
        levels = n.bit_length() - 1
        for values in config["scaling_codebooks"]:
            k = len(values)
            for query_count in config["scaling_query_counts"]:
                positions = np.linspace(
                    0, 2 ** (levels - 1) - 1, query_count, dtype=int
                )
                protected = tuple((levels - 1, int(j)) for j in positions)
                for mode in ("exact", "amplitude", "physical_mean"):
                    epsilons = (
                        config["scaling_epsilons"]
                        if mode != "exact"
                        else [config["scaling_epsilons"][0]]
                    )
                    for epsilon in epsilons:
                        started = time.perf_counter()
                        record = {
                            "time_count": n,
                            "alphabet_size": k,
                            "codebook": values,
                            "query_count": query_count,
                            "mode": mode,
                            "epsilon": epsilon,
                        }
                        try:
                            problem = compile_query_preserving_haar(
                                np.full((n, k), 1 / k),
                                np.asarray(values),
                                np.zeros(n),
                                np.full(n, 1 / n),
                                lower_bound=0,
                                offset=0,
                                epsilon=epsilon,
                                protected_nodes=(
                                    tuple(
                                        (d, j)
                                        for d in range(levels)
                                        for j in range(2**d)
                                    )
                                    if mode == "exact"
                                    else protected
                                ),
                                rounding="amplitude" if mode == "exact" else mode,
                                max_variables=budget,
                            )
                        except ValueError as exc:
                            if "budget exceeded" not in str(exc):
                                raise
                            record.update(
                                status="entry_budget_exceeded", reason=str(exc)
                            )
                        else:
                            record.update(
                                status="compiled",
                                base_variables=problem.variable_count,
                                matrix_nonzeros=problem.matrix.nnz,
                                binary_memory_variables=2**query_count
                                * problem.variable_count,
                                protected_node_count=len(problem.protected_nodes),
                                exact_table_entries=sum(
                                    t.columns.stop - t.columns.start
                                    for t in problem.tables
                                    if t.exact
                                ),
                                leaf_error_max=float(problem.leaf_error_radii.max()),
                            )
                        record["construction_seconds"] = time.perf_counter() - started
                        scaling.append(record)
    return {
        "status": "verified_query_preserving_compression",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": platform.node(),
        "config": config,
        "records": records,
        "scaling": scaling,
        "discontinuity_witnesses": witnesses,
        "oracle_max_error": max(r["oracle_max_error"] for r in records),
        "envelope_max_error": max(r["envelope_max_error"] for r in records),
        "all_true_extrema_contained": all(r["contains_true_extrema"] for r in records),
        "scaling_memory_allocated": False,
        "scaling_lp_solved": False,
        "floating_point_interval_certified": False,
        "scope": "constructed_balanced_integer_haar_full_support_singleton_marginals",
        "baseline": "same_query_skeleton_uniform_raw_amplitude_rounding",
        "same_state_graphical_solver_has_identical_feasible_class": True,
        "reference_risk_qualified": False,
    }


def evaluate_query_resampling(config, *, checkpoint_path=None):
    """Screen the full local centered-moment menu against exact exhaustive LPs."""
    records = []
    for n in config["time_counts"]:
        code = torch.tensor(config["codebook_values"], dtype=torch.float64)
        if code.tolist() != [-1.0, 0.0, 1.0]:
            raise ValueError(
                "The declared resampling families require codebook [-1,0,1]."
            )
        levels = n.bit_length() - 1
        codec = GraphSpectralTemporalWaveletTokenizer(4, 2, 8, 1.0, False)
        metadata = codec.pack_temporal_wavelets(code.new_zeros((1, n, 1, 1)))[1]
        for family in config["families"]:
            p = code.new_full((n, 3), 1 / 3)
            for depth in range(1, levels):
                active = (
                    config["activity_scale"] * 2 ** (depth / 2) / n
                    if family == "scale_decay"
                    else config["uniform_activity"]
                )
                if not 0 < active < 1:
                    raise ValueError(
                        "Declared family must have full-support marginals."
                    )
                p[2**depth : 2 ** (depth + 1)] = code.new_tensor(
                    [active / 2, 1 - active, active / 2]
                )
            if family not in ("scale_decay", "uniform_activity"):
                raise ValueError("Unknown resampling validation family.")
            started = time.perf_counter()
            problem = compile_haar_dependence_queries(
                p,
                code,
                metadata,
                code.new_zeros(n),
                code.new_full((n,), 1 / n),
                code.new_tensor(0),
                lower_bound=0,
                conditioning_key="centered_query_resampling",
                max_variables=config["max_variables"],
            )
            construction_seconds = time.perf_counter() - started
            influence = haar_resampling_influences(problem)
            if influence["influences"].sum() > influence["summability_bound"] + 1e-10:
                raise ArithmeticError(
                    "Haar influences violate their summability bound."
                )
            comparator_start = time.perf_counter()
            if family == "uniform_activity":
                risks = condition_haar_risk(
                    problem, None, conditioning_key=problem.conditioning_key
                )
                alternative = (risks["lower"] + risks["upper"]) / 2
                comparator_rule = "midpoint_of_unconditioned_attainable_risk_interval"
            else:
                alternative = config["comparator_scale"] / math.sqrt(n)
                comparator_rule = "analytic_root_decision_boundary"
            comparator_seconds = time.perf_counter() - comparator_start
            for statistic in config["statistics"]:
                for radius in config["report_radii"]:
                    if checkpoint_path is not None:
                        checkpoint_path.write_text(json.dumps({
                            "status": "running_resampling_query_menu",
                            "job_id": os.environ.get("SLURM_JOB_ID"),
                            "config": config, "records": records,
                            "active_setting": {
                                "time_count": n, "family": family,
                                "statistic": statistic, "report_radius": radius,
                            },
                        }, indent=2) + "\n")
                    costs = np.full(n - 1, config["query_cost"])
                    radii = np.full(n - 1, radius)
                    started = time.perf_counter()
                    selected = select_resampling_screened_query(
                        problem,
                        radii,
                        costs,
                        alternative_loss=alternative,
                        conditioning_key=problem.conditioning_key,
                        max_variables=config["max_paired_variables"],
                        optimality_tolerance=config["optimality_tolerance"],
                        statistic=statistic,
                    )
                    screened_seconds = time.perf_counter() - started
                    started = time.perf_counter()
                    exhaustive = []
                    for index in range(-1, n - 1):
                        ids = () if index < 0 else (index,)
                        result = worst_report_decision_regret(
                            problem,
                            ids,
                            np.empty(0) if index < 0 else np.array([radius]),
                            alternative_loss=alternative,
                            conditioning_key=problem.conditioning_key,
                            max_variables=config["max_paired_variables"],
                            statistic=statistic,
                        )
                        value = result["worst_report_regret"] + (
                            0 if index < 0 else costs[index]
                        )
                        exhaustive.append(float(value))
                    exhaustive_seconds = time.perf_counter() - started
                    best = min(exhaustive)
                    discrepancy = float(selected["selected"]["objective_upper"] - best)
                    bound_violation = float(
                        np.max(
                            np.array(selected["query_objective_lower_bounds"])
                            - exhaustive[1:]
                        )
                    )
                    if (
                        discrepancy > config["optimality_tolerance"] + 2e-8
                        or bound_violation > 2e-8
                    ):
                        raise ArithmeticError(
                            "Screening or influence bound disagrees with exhaustive menu."
                        )
                    record = {
                        "time_count": n,
                        "family": family,
                        "statistic": statistic,
                        "report_radius": radius,
                        "comparator_loss": alternative,
                        "comparator_rule": comparator_rule,
                        "comparator_setup_seconds": comparator_seconds,
                        "query_cost": config["query_cost"],
                        "base_variables": problem.variable_count,
                        "dictionary_size": n - 1,
                        "construction_seconds": construction_seconds,
                        "screened_seconds": screened_seconds,
                        "exhaustive_seconds": exhaustive_seconds,
                        "exhaustive_solve_count": n,
                        "screened_solve_count": selected["solve_count"],
                        "selected": selected,
                        "exhaustive_objectives": exhaustive,
                        "optimal_objective": best,
                        "selection_suboptimality": discrepancy,
                        "lower_bound_violation": bound_violation,
                        "all_marginals_full_support": bool((p > 0).all()),
                        "all_leaf_weights_positive": True,
                        "expected_active_nonroot_details": float((1 - p[2:, 1]).sum()),
                        "influence_sum": float(influence["influences"].sum()),
                        "analytic_loss_tail_allowance": (
                            config["activity_scale"] * (1 - 2 / n) / math.sqrt(n)
                            if family == "scale_decay"
                            else None
                        ),
                    }
                    if family == "scale_decay" and statistic == "centered_energy":
                        upper = record["analytic_loss_tail_allowance"] + radius / (
                            2 * math.sqrt(n)
                        )
                        record["analytic_root_regret_upper"] = upper
                        if exhaustive[1] > upper + 2e-8:
                            raise ArithmeticError(
                                "Root query violates independent analytic family bound."
                            )
                    records.append(record)
                    print(
                        json.dumps(
                            {
                                key: record[key]
                                for key in (
                                    "time_count",
                                    "family",
                                    "statistic",
                                    "report_radius",
                                    "screened_solve_count",
                                    "exhaustive_solve_count",
                                    "selection_suboptimality",
                                    "screened_seconds",
                                    "exhaustive_seconds",
                                )
                            }
                        ),
                        flush=True,
                    )
    return {
        "status": "verified_resampling_screened_query_menu",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": platform.node(),
        "config": config,
        "records": records,
        "max_selection_suboptimality": max(
            r["selection_suboptimality"] for r in records
        ),
        "max_lower_bound_violation": max(r["lower_bound_violation"] for r in records),
        "scope": "constructed_full_support_balanced_haar_centered_population_moment_queries",
        "full_observation_laws_preserved": False,
        "floating_point_interval_certified": False,
        "reference_risk_qualified": False,
        "baseline": "identical_centered_query_menu_exhaustive_checked_paired_LPs",
    }


def evaluate_compressed_query_resampling(config, *, checkpoint_path=None):
    """Match compressed menu search to the saved exhaustive exact objectives."""
    source = json.loads(Path(config["exact_results_path"]).read_text())
    if source["status"] != "verified_resampling_screened_query_menu":
        raise ValueError("Compressed comparison requires a complete exact-menu evaluation.")
    family_config = source["config"]
    records = []
    for exact in source["records"]:
        n = exact["time_count"]
        if n not in config["time_counts"] or exact["statistic"] not in config["statistics"]:
            continue
        code = np.asarray(family_config["codebook_values"], dtype=float)
        if code.tolist() != [-1.0, 0.0, 1.0]:
            raise ValueError("The saved full-support families require [-1,0,1].")
        p = np.full((n, 3), 1 / 3)
        for depth in range(1, n.bit_length() - 1):
            active = (
                family_config["activity_scale"] * 2 ** (depth / 2) / n
                if exact["family"] == "scale_decay" else family_config["uniform_activity"]
            )
            p[2**depth:2**(depth + 1)] = [active / 2, 1 - active, active / 2]
        if exact["family"] not in ("scale_decay", "uniform_activity"):
            raise ValueError("Unsupported exact-reference family.")
        for variant in config["variants"]:
            active_setting = {key: exact[key] for key in (
                "time_count", "family", "statistic", "report_radius",
            )}
            active_setting.update(variant)
            if checkpoint_path is not None:
                checkpoint_path.write_text(json.dumps({
                    "status": "running_compressed_resampling_comparison",
                    "job_id": os.environ.get("SLURM_JOB_ID"), "config": config,
                    "records": records, "active_setting": active_setting,
                }, indent=2) + "\n")
            started = time.perf_counter()
            result = select_compressed_resampling_query(
                p, code, np.zeros(n), np.full(n, 1 / n),
                np.full(n - 1, exact["report_radius"]), np.full(n - 1, exact["query_cost"]),
                lower_bound=0, offset=0, alternative_loss=exact["comparator_loss"],
                epsilon=config["epsilon_scale"] / math.sqrt(n),
                rounding=variant["rounding"], search=variant["search"],
                max_variables=config["max_variables"],
                max_paired_variables=config["max_paired_variables"],
                optimality_tolerance=config["optimality_tolerance_scale"] / math.sqrt(n),
                max_refinements=config["max_refinements"],
                refinement_factor=config["refinement_factor"], statistic=exact["statistic"],
            )
            elapsed = time.perf_counter() - started
            chosen = result["selected"]["option_index"]
            true_objective = exact["exhaustive_objectives"][chosen]
            error = true_objective - exact["optimal_objective"]
            lower_violation = float(np.max(
                np.asarray(result["objective_lower_bounds"]) - exact["exhaustive_objectives"]
            ))
            if (
                lower_violation > 2e-8
                or true_objective > result["selected"]["objective_upper"] + 2e-8
                or (result["selection_resolved"] and error > result["optimality_tolerance"] + 2e-8)
            ):
                raise ArithmeticError("Compressed query bounds contradict the exact menu.")
            bound = result["distinct_query_count_bound"]
            if bound is not None and len(result["solved_query_ids"]) > bound + 1e-8:
                raise ArithmeticError("Observed distinct-query count exceeds the static-order bound.")
            record = {
                **active_setting, "result": result, "total_seconds": elapsed,
                "actual_selected_objective": true_objective,
                "exact_optimal_objective": exact["optimal_objective"],
                "selection_suboptimality": error, "lower_bound_violation": lower_violation,
                "exact_paired_variables": 2 * exact["base_variables"] + 1,
                "exact_exhaustive_solve_count": exact["exhaustive_solve_count"],
                "exact_screened_solve_count": exact["screened_solve_count"],
                "exact_exhaustive_seconds": exact["exhaustive_seconds"],
                "exact_screened_seconds": exact["screened_seconds"],
                "exact_construction_seconds": exact["construction_seconds"],
                "comparator_rule": exact["comparator_rule"],
            }
            records.append(record)
            print(json.dumps({**active_setting,
                              "resolved": result["selection_resolved"],
                              "solve_count": result["solve_count"],
                              "distinct_queries": len(result["solved_query_ids"]),
                              "peak_variables": result["peak_paired_variables"],
                              "total_seconds": elapsed}), flush=True)
    expected = len(config["time_counts"]) * len(config["statistics"]) * len(
        family_config["families"]
    ) * len(family_config["report_radii"]) * len(config["variants"])
    if len(records) != expected:
        raise ValueError("Exact-menu reference does not cover every requested setting.")
    return {
        "status": "verified_compressed_resampling_comparison",
        "job_id": os.environ.get("SLURM_JOB_ID"), "node": platform.node(),
        "config": config, "records": records,
        "resolved_count": sum(r["result"]["selection_resolved"] for r in records),
        "record_count": len(records),
        "max_lower_bound_violation": max(r["lower_bound_violation"] for r in records),
        "matched_generic_screening": "identical_when_given_the_same_resampling_bounds",
        "timing_scope": "CPU_local_construction_and_LP_solve_times_single_run",
        "floating_point_interval_certified": False,
        "reference_risk_qualified": False,
        "scope": "constructed_centered_population_moment_menus_no_prior_reports",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="Libs/configs/exp/dependence_query_evaluation.yaml"
    )
    parser.add_argument("--output")
    parser.add_argument(
        "--analysis",
        choices=(
            "dependence",
            "posterior",
            "sampling",
            "observation",
            "joint",
            "compression",
            "resampling",
            "compressed_resampling",
        ),
        default="dependence",
    )
    args = parser.parse_args()
    config = load_config(args.config)["dependence_query_evaluation"]
    if args.analysis == "posterior":
        config = config["posterior_bridge"]
        result = evaluate_posterior_bridge(config)
    elif args.analysis == "sampling":
        config = config["multiscale_sampling"]
        result = evaluate_multiscale_sampling(config)
    elif args.analysis == "observation":
        config = config["observation_experiments"]
        result = evaluate_observation_experiments(config)
    elif args.analysis == "joint":
        config = config["joint_observation"]
        result = evaluate_joint_observation(config)
    elif args.analysis in ("resampling", "compressed_resampling"):
        config = config["query_resampling" if args.analysis == "resampling" else "compressed_resampling"]
        checkpoint = Path(args.output or config["output_path"])
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        try:
            evaluator = evaluate_query_resampling if args.analysis == "resampling" else evaluate_compressed_query_resampling
            result = evaluator(config, checkpoint_path=checkpoint)
        except Exception as exc:
            partial = json.loads(checkpoint.read_text()) if checkpoint.exists() else {
                "records": [], "config": config,
            }
            partial.update(status="failed_resampling_query_menu", failure=str(exc),
                           failure_type=type(exc).__name__)
            checkpoint.write_text(json.dumps(partial, indent=2) + "\n")
            raise
    elif args.analysis == "compression":
        config = config["query_compression"]
        result = evaluate_query_compression(config)
    else:
        result = run_evaluation(config)
    output = Path(args.output or config["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
