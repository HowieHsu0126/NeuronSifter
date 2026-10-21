"""Read-only, open-world optimization and checkpoint-identity adjudication.

This audit does not select weights, fit a model/calibrator, grant independent
approval, or read a sealed payload. Convergence and scientific utility are
separate from the integrity of the frozen selected candidate.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from contextlib import redirect_stdout
from datetime import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import pickle
import random
import statistics
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from Libs.exp.training.train_operational_twin_devcal import (
    _action_encoding_for_contract,
    _normalized,
    build_method_model,
)
from Libs.models.modules.world_action_encoder import WorldActionEncoder
from Libs.utils.exp.experiment_manifests import native_training_seed_values
from Libs.utils.data.controlled_world_contracts import FAMILY_UIDS
from Libs.utils.data.controlled_world_tensorization import (
    ControlledWorldTensorDataset,
    tensorize_manifest,
)
from Libs.utils.evaluation.operational_twin_development_evidence import (
    FULL_METHOD_ID,
    LOWER_IS_BETTER_METRICS,
    PRIMARY_BASELINE_METHODS,
    build_development_comparisons,
    training_budget_evidence,
)
from Libs.utils.evaluation.operational_twin_training_evidence_ledger import (
    validate_operational_twin_training_evidence_ledger,
)
from Libs.utils.evaluation.operational_twin_training_protocol import (
    REQUIRED_METHOD_IDS,
    resolve_method_execution_contract,
)

SCHEMA_VERSION = "neuronsifter-operational-twin-training-sufficiency-v1"
POLICY_PATH = "Libs/configs/validation/operational_twin_training_sufficiency.yaml"
SCHEMA_PATH = "Libs/configs/schema/operational_twin_training_sufficiency.schema.json"
MANIFEST = "Output/results/operational_twin/training_protocol/assignment_manifest.json"
LEDGER = "Output/audits/operational_twin_training_evidence_ledger.json"
CANDIDATE = "Output/audits/operational_twin_sufficient_training_candidate.json"
OPEN_BASE = "Input/processed/controlled_neural_world/open_action_identification/"
UNOPENED_RECEIPTS = (
    "Input/processed/controlled_neural_world/benchmark/benchmark_freeze_receipt.json",
    "Input/processed/controlled_neural_world/benchmark/access_receipts/twin_qualification_unopened_access_receipt.json",
)
STATES = (
    "SUFFICIENTLY_TRAINED",
    "CLEARLY_UNDERTRAINED",
    "NONCONVERGENT",
    "NUMERICALLY_INVALID",
    "INDETERMINATE_WITHIN_FROZEN_BUDGET",
)
CLOSED = {
    "sufficient_training": False,
    "twin_qualification_pass": False,
    "formal_qualification": False,
    "sealed_test_access": 0,
    "protected_holdout_access": 0,
    "six_context_outcome_access": 0,
    "agent_trace_gate_open": False,
    "agent_work_open": False,
    "tool_gate_open": False,
    "memory_gate_open": False,
    "coevolution_gate_open": False,
}


# These identities preserve observations predating the explicit runtime profile.
CRITERION_FREEZE_COMMIT = "4ad5577"
PRE_PROFILE_AUDIT_COMMIT = "417623382420a51ef3701d427471270d7f827e15"
PROFILE_REVIEW_COMMIT = "2c6fb750a7cc79d37976310715d3b241040fc584"
PROFILE_REVIEW_PATH = (
    "orchestrator/rounds/round-38-freeze-sufficient-twin-before-sealed-access/review.md"
)
# Capture startup settings before callers can try to normalize only os.environ.
_CPU_ENVIRONMENT_AT_IMPORT = {
    key: os.environ.get(key)
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "OMP_DYNAMIC",
        "MKL_DYNAMIC",
        "PYTHONHASHSEED",
        "MKL_CBWR",
        "ATEN_CPU_CAPABILITY",
        "ONEDNN_MAX_CPU_ISA",
        "DNNL_MAX_CPU_ISA",
    )
}


def validate_cpu_audit_profile(profile: Mapping) -> None:
    """One supported audit execution identity, not a new tuning surface."""
    expected = {
        "profile_id": "single_thread_cpu_exact_replay",
        "torch_intra_op_threads": 1,
        "torch_inter_op_threads": 1,
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "mkldnn_enabled": True,
        "mkldnn_deterministic": True,
        "fp32_precision": "ieee",
        "environment": {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
            "MKL_DYNAMIC": "FALSE",
            "PYTHONHASHSEED": "0",
            "MKL_CBWR": None,
            "ATEN_CPU_CAPABILITY": None,
            "ONEDNN_MAX_CPU_ISA": None,
            "DNNL_MAX_CPU_ISA": None,
        },
    }
    if set(profile) != {*expected, "rationale"} or not profile.get("rationale"):
        raise ValueError("incomplete CPU audit execution profile")
    if any(
        type(profile[k]) is not type(v) or profile[k] != v for k, v in expected.items()
    ):
        raise ValueError("unsupported CPU audit execution profile")


def bootstrap_cpu_audit_cli(policy: Mapping) -> None:
    """Restart the CLI before numerical work so native startup settings are real.

    Changing os.environ after importing Torch/NumPy alone does not reinitialize
    native thread pools. The fresh interpreter is essential, not a fallback.
    Original ambient settings are reported but never mixed into canonical values.
    """
    profile = policy["audit_cpu_execution"]
    validate_cpu_audit_profile(profile)
    environment = profile["environment"]
    ambient = {key: os.environ.get(key) for key in environment}
    if ambient == environment and _CPU_ENVIRONMENT_AT_IMPORT == environment:
        return
    clean_environment = dict(os.environ)
    for key, value in environment.items():
        if value is None:
            clean_environment.pop(key, None)
        else:
            clean_environment[key] = value
    print(
        json.dumps(
            {
                "cpu_audit_environment_restart": True,
                "ambient": ambient,
                "effective_requested": environment,
            }
        ),
        flush=True,
    )
    os.execve(sys.executable, [sys.executable, *sys.orig_argv[1:]], clean_environment)


def configure_cpu_audit_execution(policy: Mapping) -> dict:
    """Enforce and measure the actual profile before any candidate replay."""
    profile = policy["audit_cpu_execution"]
    validate_cpu_audit_profile(profile)
    if (
        _CPU_ENVIRONMENT_AT_IMPORT != profile["environment"]
        or {key: os.environ.get(key) for key in profile["environment"]}
        != profile["environment"]
    ):
        raise ValueError("CPU audit requires the fresh-interpreter CLI environment")
    torch.set_num_threads(profile["torch_intra_op_threads"])
    if torch.get_num_interop_threads() != profile["torch_inter_op_threads"]:
        try:
            torch.set_num_interop_threads(profile["torch_inter_op_threads"])
        except RuntimeError as error:
            raise ValueError(
                "CPU inter-op pool already initialized; use a fresh audit CLI"
            ) from error
    torch.use_deterministic_algorithms(
        profile["deterministic_algorithms"],
        warn_only=profile["deterministic_warn_only"],
    )
    torch.backends.mkldnn.enabled = profile["mkldnn_enabled"]
    torch.backends.mkldnn.deterministic = profile["mkldnn_deterministic"]
    torch.backends.fp32_precision = profile["fp32_precision"]
    torch.backends.mkldnn.fp32_precision = profile["fp32_precision"]
    if (
        torch.get_default_dtype() != torch.float32
        or str(torch.get_default_device()) != "cpu"
        or torch.is_autocast_enabled("cpu")
    ):
        raise ValueError("CPU audit requires float32 CPU defaults without autocast")
    actual = {
        "profile_id": profile["profile_id"],
        "torch_intra_op_threads": torch.get_num_threads(),
        "torch_inter_op_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "mkldnn_enabled": torch.backends.mkldnn.enabled,
        "mkldnn_deterministic": torch.backends.mkldnn.deterministic,
        "fp32_precision": torch.backends.fp32_precision,
        "environment": {key: os.environ.get(key) for key in profile["environment"]},
    }
    if actual != {key: value for key, value in profile.items() if key != "rationale"}:
        raise ValueError(
            "effective CPU audit settings disagree with the frozen profile"
        )
    if torch.backends.mkldnn.fp32_precision != profile["fp32_precision"]:
        raise ValueError("effective MKLDNN precision disagrees with the frozen profile")
    numpy_build = io.StringIO()
    with redirect_stdout(numpy_build):
        np.__config__.show()
    return {
        **actual,
        "device": "cpu",
        "default_dtype": str(torch.get_default_dtype()),
        "cpu_autocast_enabled": torch.is_autocast_enabled("cpu"),
        "mkldnn_fp32_precision": torch.backends.mkldnn.fp32_precision,
        "host": platform.node(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "numpy_build_config": numpy_build.getvalue(),
        "torch_version": str(torch.__version__),
        "torch_git_version": torch.version.git_version,
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "torch_build_config": torch.__config__.show(),
        "torch_parallel_info": torch.__config__.parallel_info(),
    }


def historical_replay_evidence(root: Path, policy: Mapping) -> dict:
    """Retain pre-profile numerical observations without treating them as a profile."""

    def historical(commit: str, path: str) -> tuple[dict, bytes]:
        # These three Git-only reads name fixed policy/audit/review metadata;
        # the review directory name is not a request to read a sealed payload.
        data = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=root)
        full_commit = (
            subprocess.check_output(["git", "rev-parse", commit], cwd=root)
            .decode()
            .strip()
        )
        return {
            "commit": full_commit,
            "path": path,
            "sha256": hashlib.sha256(data).hexdigest(),
        }, data

    criterion_source, criterion_bytes = historical(CRITERION_FREEZE_COMMIT, POLICY_PATH)
    original_policy = yaml.safe_load(criterion_bytes)[
        "operational_twin_training_sufficiency"
    ]
    if {
        key: value for key, value in policy.items() if key != "audit_cpu_execution"
    } != original_policy:
        raise ValueError(
            "original frozen sufficiency criteria changed beyond audit execution settings"
        )
    source, payload = historical(
        PRE_PROFILE_AUDIT_COMMIT, policy["outputs"]["per_seed"]
    )
    previous = json.loads(payload)
    mismatches = [
        {
            "assignment_uid": row["assignment_uid"],
            "method_id": row["method_id"],
            "seed": row["seed"],
            **mismatch,
        }
        for row in previous["assignments"]
        for mismatch in row["checkpoint_identity"]
        .get("replay", {})
        .get("archived_metric_mismatches", [])
    ]
    review_source, review_bytes = historical(PROFILE_REVIEW_COMMIT, PROFILE_REVIEW_PATH)
    return {
        "optimization_criterion_freeze": criterion_source,
        "pre_profile_audit": {
            "source": source,
            "runtime_profile_status": "UNRECORDED_NOT_INFERRED_FROM_CPU_ALLOCATION",
            "archived_metric_mismatches": mismatches,
        },
        "independent_ambient_thread_review": {
            "source": review_source,
            "document": review_bytes.decode(),
        },
        "interpretation": "Historical observations are retained, not combined into or substituted for canonical profile replay.",
    }


def safe_path(root: Path, relative: str) -> Path:
    """Deny escaped/symlinked/held-out inputs before reading any bytes."""
    p = Path(relative)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError("noncanonical source path")
    allowed_metadata = relative in UNOPENED_RECEIPTS
    if not allowed_metadata and any(
        token in relative.lower()
        for token in (
            "sealed",
            "test_manifest",
            "protected",
            "six_context",
            "private",
            "benchmark/test",
            "ood_manifest",
        )
    ):
        raise ValueError("denied held-out source path")
    for parent in (p, *p.parents):
        if (root / parent).is_symlink():
            raise ValueError("mutable symlink source is not admissible")
    if not (root / p).resolve().is_relative_to(root.resolve()):
        raise ValueError("source escaped root")
    return root / p


def source_bytes(root: Path, relative: str, sources: dict) -> bytes:
    p = safe_path(root, relative)
    data = p.read_bytes()
    sources[relative] = {
        "path": relative,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }
    return data


def read_json(root: Path, relative: str, sources: dict):
    return json.loads(source_bytes(root, relative, sources))


def read_rows(root: Path, relative: str, sources: dict) -> list[dict]:
    return [
        json.loads(line)
        for line in source_bytes(root, relative, sources).decode().splitlines()
        if line
    ]


def finite_tree(value) -> bool:
    if isinstance(value, Mapping):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite_tree(v) for v in value)
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    return not isinstance(value, float) or math.isfinite(value)


def late_diagnostic(values: Sequence[float], policy: Mapping) -> dict:
    curve = np.asarray(values, dtype=float)
    if not np.isfinite(curve).all():
        raise ValueError("nonfinite optimization curve")
    n = max(
        int(policy["minimum_late_epochs"]),
        math.ceil(len(curve) * float(policy["late_fraction"])),
    )
    if len(curve) < n:
        raise ValueError("too few epochs for the frozen late window")
    late = curve[-n:]
    scale = max(
        float(np.quantile(curve, 0.9) - np.quantile(curve, 0.1)),
        abs(float(np.median(late))) * float(policy["normalization_level_fraction"]),
        float(policy["absolute_scale_floor"]),
    )
    slope = statistics.median(
        (float(late[j]) - float(late[i])) / (j - i)
        for i in range(n)
        for j in range(i + 1, n)
    )
    residual = late - slope * np.arange(n)
    return {
        "epoch_count": len(curve),
        "late_epoch_count": n,
        "first": float(curve[0]),
        "last": float(curve[-1]),
        "minimum": float(curve.min()),
        "minimum_epoch": int(curve.argmin()) + 1,
        "normalization_scale": scale,
        "median_pairwise_slope": slope,
        "normalized_late_change": slope * (n - 1) / scale,
        "normalized_residual_iqr": float(
            np.quantile(residual, 0.75) - np.quantile(residual, 0.25)
        )
        / scale,
    }


def classify_optimization(
    train: Mapping | None,
    dev: Mapping | None,
    policy: Mapping,
    *,
    numerical_valid: bool = True,
    mandatory_complete: bool = True,
    no_fit: bool = False,
) -> tuple[str, str]:
    if not numerical_valid:
        return STATES[3], "nonfinite_numerical_evidence"
    if not mandatory_complete:
        return STATES[4], "missing_or_inconsistent_mandatory_optimization_evidence"
    if no_fit:
        return STATES[0], "explicit_nonoptimizing_persistence_contract_satisfied"
    if train is None or dev is None:
        return STATES[4], "matched_training_and_development_objective_required"
    t, d = train["normalized_late_change"], dev["normalized_late_change"]
    if (
        t > policy["nonconvergent_min_normalized_worsening"]
        and train["normalized_residual_iqr"] > policy["nonconvergent_min_residual_iqr"]
    ):
        return STATES[2], "worsening_and_oscillating_training_objective"
    if (
        t < -policy["clear_improvement_min_normalized_change"]
        and d < -policy["clear_improvement_min_normalized_change"]
    ):
        return STATES[1], "training_and_matched_development_objective_still_improving"
    if (
        abs(t) <= policy["plateau_max_normalized_change"]
        and abs(d) <= policy["plateau_max_normalized_change"]
    ):
        return STATES[0], "training_and_matched_development_objectives_plateau"
    return STATES[4], "ambiguous_or_discordant_late_optimization"


def aggregate_methods(
    rows: Sequence[Mapping], methods: Sequence[str], seeds: Sequence[int]
) -> list[dict]:
    expected = {(m, s) for m in methods for s in seeds}
    observed = [(r["method_id"], r["seed"]) for r in rows]
    if set(observed) != expected or len(observed) != len(expected):
        raise ValueError("all fixed method/seed assignments required exactly once")
    result = []
    for method in methods:
        selected = [r for r in rows if r["method_id"] == method]
        states = [r["classification"] for r in selected]
        # All-seed proof is conjunctive; mixed terminal evidence is retained.
        status = states[0] if len(set(states)) == 1 else STATES[4]
        result.append(
            {
                "method_id": method,
                "seeds": list(seeds),
                "classification": status,
                "per_seed_classifications": {
                    str(r["seed"]): r["classification"] for r in selected
                },
                "identity_blocker_count": sum(
                    len(r["identity_blockers"]) for r in selected
                ),
                "all_seeds_optimization_sufficient": all(
                    s == STATES[0] for s in states
                ),
            }
        )
    return result


def independent_pair_metrics(
    worlds: Sequence[Mapping], pairs: Sequence[Mapping]
) -> list[dict]:
    by_uid = {r["world_uid"]: r for r in worlds}
    required = [u for p in pairs for u in p["member_world_uids"]]
    if len(by_uid) != len(worlds) or set(by_uid) != set(required):
        raise ValueError("missing/duplicate/extra open world")
    result = []
    for pair in pairs:
        if pair.get("approved_pair") is not True or len(pair["member_world_uids"]) != 2:
            raise ValueError("unapproved/incomplete counterfactual pair")
        left, right = [by_uid[u] for u in pair["member_world_uids"]]
        if (
            left["family_uid"] != right["family_uid"]
            or left["family_uid"] != pair["family_uid"]
        ):
            raise ValueError("pair family mismatch")
        values = [
            float(r[k])
            for r in (left, right)
            for k in ("reference_utility", "predicted_utility", "trajectory_rmse")
        ]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("nonfinite retained world outcome")
        truth = right["reference_utility"] - left["reference_utility"]
        prediction = right["predicted_utility"] - left["predicted_utility"]
        chosen = right if prediction > 0 else left  # Frozen first-arm tie rule.
        result.append(
            {
                "method_id": left["method_id"],
                "seed": left["seed"],
                "counterfactual_pair_uid": pair["counterfactual_pair_uid"],
                "family_uid": pair["family_uid"],
                "action_contrast_error": abs(prediction - truth),
                "intervention_ranking_regret": max(
                    left["reference_utility"], right["reference_utility"]
                )
                - chosen["reference_utility"],
                "effect_direction_error": None
                if truth == 0
                else float((truth > 0) != (prediction > 0)),
                "trajectory_rmse": (left["trajectory_rmse"] + right["trajectory_rmse"])
                / 2,
            }
        )
    return result


def independent_statistics(
    pairs: Sequence[Mapping], seeds: Sequence[int], policy: Mapping
) -> list[dict]:
    """Independent numeric-array replay of the frozen world-level statistics."""
    keyed = {
        (r["method_id"], r["seed"], r["counterfactual_pair_uid"]): r for r in pairs
    }
    pair_ids = sorted({r["counterfactual_pair_uid"] for r in pairs})
    comparisons = []
    for baseline in PRIMARY_BASELINE_METHODS:
        for metric in LOWER_IS_BETTER_METRICS:
            differences, families, undefined = [], [], 0
            for uid in pair_ids:
                per_seed = []
                for seed in seeds:
                    left = keyed[(baseline, seed, uid)][metric]
                    right = keyed[(FULL_METHOD_ID, seed, uid)][metric]
                    if left is None or right is None:
                        undefined += 1
                    else:
                        per_seed.append(left - right)
                if per_seed and len(per_seed) != len(seeds):
                    raise ValueError("partial seed denominator is forbidden")
                if per_seed:
                    differences.append(statistics.fmean(per_seed))
                    families.append(
                        keyed[(FULL_METHOD_ID, seeds[0], uid)]["family_uid"]
                    )
            if not differences:
                comparisons.append(
                    {
                        "baseline": baseline,
                        "metric": metric,
                        "status": "UNDEFINED",
                        "undefined_seed_pairs": undefined,
                    }
                )
                continue
            strata = [
                [v for v, f in zip(differences, families) if f == family]
                for family in sorted(set(families))
            ]
            if len(strata) != len(FAMILY_UIDS):
                raise ValueError("missing family stratum")
            estimate = statistics.fmean(statistics.fmean(v) for v in strata)
            rng = random.Random(policy["analysis_seed"])
            draws = sorted(
                statistics.fmean(
                    statistics.fmean(group[rng.randrange(len(group))] for _ in group)
                    for group in strata
                )
                for _ in range(policy["bootstrap"]["resamples"])
            )
            alpha = 1 - policy["bootstrap"]["confidence_level"]
            lower, upper = (
                draws[int(alpha / 2 * (len(draws) - 1))],
                draws[int((1 - alpha / 2) * (len(draws) - 1))],
            )
            rng = random.Random(policy["analysis_seed"] + 1)
            exceed = 0
            for _ in range(policy["conditional_permutation"]["permutations"]):
                signed = [v * (-1 if rng.random() < 0.5 else 1) for v in differences]
                statistic = statistics.fmean(
                    statistics.fmean(v for v, f in zip(signed, families) if f == family)
                    for family in sorted(set(families))
                )
                exceed += abs(statistic) >= abs(estimate)
            comparisons.append(
                {
                    "baseline": baseline,
                    "metric": metric,
                    "status": "DEFINED",
                    "estimate": estimate,
                    "lower": lower,
                    "upper": upper,
                    "p_value": (exceed + 1)
                    / (policy["conditional_permutation"]["permutations"] + 1),
                    "world_count": len(differences),
                    "seed_count": len(seeds),
                    "undefined_seed_pairs": undefined,
                    "families": {
                        f: statistics.fmean(
                            v for v, g in zip(differences, families) if g == f
                        )
                        for f in sorted(set(families))
                    },
                }
            )
    defined = [r for r in comparisons if r["status"] == "DEFINED"]
    running = 0.0
    for rank, row in enumerate(sorted(defined, key=lambda r: r["p_value"])):
        running = max(running, min(1.0, row["p_value"] * (len(defined) - rank)))
        row["holm_adjusted_p_value"] = running
    return comparisons


def _tensor_equal(left, right) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, Mapping):
        return (
            isinstance(right, Mapping)
            and left.keys() == right.keys()
            and all(_tensor_equal(v, right[k]) for k, v in left.items())
        )
    if isinstance(left, (tuple, list)):
        return (
            isinstance(right, (tuple, list))
            and len(left) == len(right)
            and all(_tensor_equal(a, b) for a, b in zip(left, right))
        )
    return left == right


@torch.no_grad()
def replay_checkpoint(
    checkpoint: Mapping,
    dataset: ControlledWorldTensorDataset,
    assignment: Mapping,
    archived_worlds: Sequence[Mapping],
    tolerance: Mapping,
    role: str = "development",
) -> dict:
    """Non-fitting CPU replay on every declared open development world."""
    if role not in ("development", "calibration") or any(
        r["role"] != role for r in archived_worlds
    ):
        raise ValueError("replay role mismatch")
    contract = resolve_method_execution_contract(assignment["method_id"])
    model = build_method_model(
        len(checkpoint["feature_names"]),
        WorldActionEncoder.output_dim,
        checkpoint["model_config"],
        contract,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    recorded = {r["world_uid"]: r for r in archived_worlds}
    errors, replay_rows = [], []
    norm_mean, norm_std = (
        checkpoint["normalization_mean"],
        checkpoint["normalization_std"],
    )
    if not finite_tree(checkpoint) or not bool((norm_std > 0).all()):
        raise ValueError("invalid checkpoint numerical state")
    for batch in DataLoader(
        dataset,
        batch_size=checkpoint["method_protocol"]["budget"]["batch_size"],
        shuffle=False,
    ):
        inputs, input_mask, targets, masks = _normalized(
            batch, norm_mean, norm_std, torch.device("cpu")
        )
        output = model(
            inputs,
            input_mask,
            _action_encoding_for_contract(batch, contract),
            batch["family_index"],
        )
        for i, uid in enumerate(batch["world_uid"]):
            residual = (output["predictive_mean"][i] - targets[i])[masks[i]]
            errors.append(residual)
            row = {
                "world_uid": uid,
                "trajectory_rmse": float(residual.square().mean().sqrt()),
            }
            if contract["calibration_eligible"]:
                scale = output["predictive_scale"][i][masks[i]]
                if role == "calibration":
                    scale = (
                        scale
                        * checkpoint["calibration_scale"][int(batch["family_index"][i])]
                    )
                row.update(
                    gaussian_nll=float(
                        (0.5 * (residual / scale).square() + scale.log()).mean()
                    ),
                    coverage_95=float((residual.abs() <= 1.96 * scale).float().mean()),
                )
            replay_rows.append(row)
    mismatches = []
    for row in replay_rows:
        for key in sorted(row.keys() - {"world_uid"}):
            if not math.isclose(
                row[key],
                recorded[row["world_uid"]][key],
                rel_tol=tolerance["relative_tolerance"],
                abs_tol=tolerance["absolute_tolerance"],
            ):
                mismatches.append(
                    {
                        "world_uid": row["world_uid"],
                        "metric": key,
                        "replayed": row[key],
                        "recorded": recorded[row["world_uid"]][key],
                    }
                )
    return {
        "reconstruction_rmse": float(torch.cat(errors).square().mean().sqrt()),
        "world_count": len(replay_rows),
        "archived_metric_mismatches": mismatches,
        "method_runtime_loaded": True,
        "calibration_refit": False,
        "role": role,
        "calibration_applied": role == "calibration"
        and contract["calibration_eligible"],
        "device": "cpu",
    }


def checkpoint_identity(
    root: Path,
    assignment: Mapping,
    ledger_row: Mapping,
    curve: Sequence[Mapping],
    worlds: Sequence[Mapping],
    dataset: ControlledWorldTensorDataset,
    sources: dict,
    policy: Mapping,
) -> tuple[dict, list[str]]:
    ref = ledger_row["checkpoint"]
    if ref["path"] is None:
        return {"status": "NOT_APPLICABLE_NO_FIT", "checkpoint_created": False}, []
    path = safe_path(root, ref["path"])
    source_bytes(root, ref["path"], sources)
    problems = []
    if sources[ref["path"]]["sha256"] != ref["sha256"]:
        problems.append("checkpoint_hash_mismatch")
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if (
        ck["assignment_uid"] != assignment["assignment_uid"]
        or ck["seed"] != assignment["seed"]
    ):
        problems.append("checkpoint_assignment_identity_mismatch")
    if ck["method_contract"] != resolve_method_execution_contract(
        assignment["method_id"]
    ):
        problems.append("checkpoint_method_contract_mismatch")
    if ck["epoch"] != min(curve, key=lambda row: row["reconstruction_rmse"])["epoch"]:
        problems.append("checkpoint_selection_rule_mismatch")
    steps = sorted(
        {
            int(s["step"])
            for s in ck["optimizer_state_dict"]["state"].values()
            if "step" in s
        }
    )
    if not steps or max(steps) > ck["step"]:
        problems.append("optimizer_state_disagrees_with_labeled_checkpoint_step")
    lrs = sorted({float(g["lr"]) for g in ck["optimizer_state_dict"]["param_groups"]})
    if lrs != [ck["method_protocol"]["budget"]["learning_rate"]]:
        problems.append("checkpoint_learning_rate_mismatch")
    replay = replay_checkpoint(
        ck, dataset, assignment, worlds, policy["checkpoint_replay"]
    )
    recorded = float(ck["dev_metrics"]["reconstruction_rmse"])

    def close(a, b):
        return math.isclose(
            a,
            b,
            rel_tol=policy["checkpoint_replay"]["relative_tolerance"],
            abs_tol=policy["checkpoint_replay"]["absolute_tolerance"],
        )

    selected_match = close(replay["reconstruction_rmse"], recorded)
    final_match = close(replay["reconstruction_rmse"], curve[-1]["reconstruction_rmse"])
    if not selected_match:
        problems.append("selected_epoch_metric_not_reproduced")
    if replay["archived_metric_mismatches"]:
        problems.append("archived_development_metrics_not_reproduced")
    last = path.with_name("last.pt")
    last_info = {"present": False}
    if last.is_file():
        relative = last.relative_to(root).as_posix()
        source_bytes(root, relative, sources)
        lc = torch.load(last, map_location="cpu", weights_only=True)
        last_info = {
            "present": True,
            "path": relative,
            "epoch": lc["epoch"],
            "step": lc["step"],
            "all_model_tensors_equal": _tensor_equal(
                ck["model_state_dict"], lc["model_state_dict"]
            ),
            "all_optimizer_state_equal": _tensor_equal(
                ck["optimizer_state_dict"], lc["optimizer_state_dict"]
            ),
        }
        if (
            last_info["all_model_tensors_equal"]
            and lc["epoch"] != ck["epoch"]
            and not selected_match
        ):
            problems.append("best_checkpoint_aliases_final_weights")
    identity = {
        "status": "PASS" if not problems else "SELECTION_IDENTITY_INVALID",
        "checkpoint_created": True,
        "sha256": ref["sha256"],
        "labeled_epoch": ck["epoch"],
        "labeled_step": ck["step"],
        "optimizer_steps": steps,
        "learning_rates": lrs,
        "numerical_state_finite": finite_tree(ck),
        "selected_recorded_rmse": recorded,
        "last_recorded_rmse": curve[-1]["reconstruction_rmse"],
        "selected_metric_reproduced": selected_match,
        "last_metric_reproduced": final_match,
        "last_checkpoint_comparison": last_info,
        "replay": replay,
        "recoverability": "declared_best_is_selected_epoch"
        if selected_match and not problems
        else "genuine_selected_epoch_weights_not_identified_in_declared_retained_checkpoints",
    }
    return identity, problems


def diagnose_assignment(
    assignment: Mapping,
    method: Mapping,
    status: Mapping,
    rows: Sequence[Mapping],
    family_weights: Mapping[str, int],
    policy: Mapping,
) -> dict:
    nofit = assignment["runtime_contract"]["checkpoint_creation_allowed"] is False
    issues = []
    budget = training_budget_evidence(status, method["budget"], rows)
    finite = finite_tree(rows)
    train = dev = None
    if nofit:
        complete = (
            status.get("status") == "COMPLETED"
            and status.get("checkpoint_created") is False
            and status.get("calibration_fitting_executed") is False
            and budget["full_budget_completed"]
        )
    else:
        epochs = [r.get("epoch") for r in rows]
        if any(type(e) is not int for e in epochs) or epochs != list(
            range(1, len(rows) + 1)
        ):
            issues.append("malformed_missing_duplicate_or_noncontiguous_epochs")
        if any(
            r.get("loss_contract") != assignment["runtime_contract"]["loss_name"]
            for r in rows
        ):
            issues.append("wrong_loss_semantics")
        if any(set(r.get("family_metrics", {})) != set(FAMILY_UIDS) for r in rows):
            issues.append("family_curve_incomplete")
        if not budget["full_budget_completed"]:
            issues.append("frozen_budget_not_completed")
        if finite and not issues:
            try:
                train = late_diagnostic(
                    [r["training_nll"] for r in rows], policy["curves"]
                )
                if method["calibration_eligible"]:
                    dev_values = [
                        sum(
                            r["family_metrics"][f]["gaussian_nll"] * family_weights[f]
                            for f in FAMILY_UIDS
                        )
                        / sum(family_weights.values())
                        for r in rows
                    ]
                else:
                    dev_values = [r["reconstruction_rmse"] ** 2 for r in rows]
                dev = late_diagnostic(dev_values, policy["curves"])
            except (KeyError, TypeError, ValueError) as error:
                issues.append("matched_objective_unavailable: " + str(error))
        complete = not issues
    classification, reason = classify_optimization(
        train,
        dev,
        policy["curves"],
        numerical_valid=finite,
        mandatory_complete=complete,
        no_fit=nofit,
    )
    timestamps = {k: status.get(k) for k in ("started_at", "finished_at")}
    elapsed = None
    if all(timestamps.values()):
        elapsed = (
            datetime.fromisoformat(timestamps["finished_at"])
            - datetime.fromisoformat(timestamps["started_at"])
        ).total_seconds()
    return {
        "assignment_uid": assignment["assignment_uid"],
        "method_id": assignment["method_id"],
        "seed": assignment["seed"],
        "classification": classification,
        "classification_reason": reason,
        "optimization_evidence_issues": issues,
        "budget": budget,
        "training_objective": train,
        "matched_development_objective": dev,
        "training_objective_semantics": assignment["runtime_contract"]["loss_name"],
        "dev_objective_semantics": policy["curves"]["dev_nll_weighting"]
        if method["calibration_eligible"]
        else policy["curves"]["dev_mse_weighting"],
        "elapsed_seconds": elapsed,
        "numerical_curve_finite": finite,
        "overfit_signal": bool(
            train
            and dev
            and train["normalized_late_change"] < 0 < dev["normalized_late_change"]
        ),
        "optional_unavailable": [
            "per_step_gradient_norm_series",
            "intermediate_checkpoint_series",
            "per_epoch_calibration_series",
        ],
        "optimizer_schedule": "constant_lr_adamw_no_scheduler"
        if not nofit
        else "not_applicable",
        "action_sensitivity_first_last": None
        if nofit
        else [
            rows[0].get("action_contrast_mean"),
            rows[-1].get("action_contrast_mean"),
        ],
        "uncalibrated_coverage_first_last": None
        if not method["calibration_eligible"]
        else [rows[0]["coverage_95"], rows[-1]["coverage_95"]],
    }


def validate_policy(policy: Mapping) -> None:
    """Validate the one supported frozen audit policy, never silently coerce it."""
    validate_cpu_audit_profile(policy["audit_cpu_execution"])
    c = policy["curves"]
    if (
        not 0 < c["late_fraction"] <= 1
        or type(c["minimum_late_epochs"]) is not int
        or c["minimum_late_epochs"] < 3
    ):
        raise ValueError("invalid late-window policy")
    for key in (
        "normalization_level_fraction",
        "absolute_scale_floor",
        "plateau_max_normalized_change",
        "clear_improvement_min_normalized_change",
        "nonconvergent_min_residual_iqr",
        "nonconvergent_min_normalized_worsening",
    ):
        if not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError("invalid convergence tolerance: " + key)
    if (
        c["plateau_max_normalized_change"]
        >= c["clear_improvement_min_normalized_change"]
    ):
        raise ValueError("plateau and continuing-improvement bands must be separated")
    if (
        c["slope_estimator"] != "median_pairwise"
        or c["dev_mse_weighting"] != "valid_element_pooled_mse"
        or c["dev_nll_weighting"] != "valid_element_weighted_family_nll"
        or policy["checkpoint_replay"]["dataset_role"] != "development"
        or policy["checkpoint_replay"]["refit_calibration"] is not False
        or policy["classification"]["method_aggregation"] != "all_three_frozen_seeds"
    ):
        raise ValueError("unsupported objective, replay, or aggregation semantics")
    if (
        policy["independent_review_required"] is not True
        or policy["audit_only"] is not True
        or policy["continuation_authorized"] is not False
    ):
        raise ValueError("audit cannot grant independent approval or continuation")


def build_audit(root: Path, config: Mapping) -> dict[str, dict]:
    policy = config["operational_twin_training_sufficiency"]
    validate_policy(policy)
    execution = configure_cpu_audit_execution(policy)
    history = historical_replay_evidence(root, policy)
    if (
        policy["audit_only"] is not True
        or policy["continuation_authorized"] is not False
    ):
        raise ValueError("this path cannot authorize fitting or sealed access")
    sources = {}
    policy_bytes = source_bytes(root, POLICY_PATH, sources)
    committed_policy = subprocess.check_output(
        ["git", "show", "HEAD:" + POLICY_PATH], cwd=root
    )
    if (
        policy_bytes != committed_policy
        or yaml.safe_load(policy_bytes)["operational_twin_training_sufficiency"]
        != policy
    ):
        raise ValueError("sufficiency policy is uncommitted or overridden after freeze")
    for path in (
        SCHEMA_PATH,
        "Libs/utils/evaluation/operational_twin_training_sufficiency.py",
        "Libs/scripts/builders/build_operational_twin_training_sufficiency.py",
        "Libs/scripts/validators/validate_operational_twin_training_sufficiency.py",
    ):
        source_bytes(root, path, sources)
    manifest = read_json(root, MANIFEST, sources)
    ledger = read_json(root, LEDGER, sources)
    candidate = read_json(root, CANDIDATE, sources)
    protocol = config["operational_twin_training_protocol"]
    if (
        tuple(manifest["method_ids"]) != tuple(REQUIRED_METHOD_IDS)
        or manifest["seeds"] != protocol["seeds"]
    ):
        raise ValueError("frozen assignment matrix mismatch")
    errors = validate_operational_twin_training_evidence_ledger(ledger, protocol)
    if errors:
        raise ValueError("invalid training ledger: " + str(errors))
    for dep in ledger["dependencies"]:
        payload = source_bytes(root, dep["path"], sources)
        if hashlib.sha256(payload).hexdigest() != dep["sha256"]:
            # New audit policy/schema includes do not rewrite original run configs.
            if dep["path"] == "Libs/configs/config_stack.yaml":
                old = b"\n".join(
                    line
                    for line in payload.split(b"\n")
                    if b"operational_twin_training_sufficiency" not in line
                )
                if hashlib.sha256(old).hexdigest() == dep["sha256"]:
                    continue
            raise ValueError("frozen dependency drift: " + dep["path"])
    frozen, unopened = [read_json(root, p, sources) for p in UNOPENED_RECEIPTS]
    if (
        unopened["result_opened"] is not False
        or unopened["formal_result_access_count"] != 0
        or any(unopened["access_counters"].values())
        or frozen["freeze_uid"] != unopened["freeze_uid"]
    ):
        raise ValueError("sealed partition is not unopened")
    source_bytes(root, OPEN_BASE + "development_manifest.json", sources)
    tensorized = tensorize_manifest(
        root / (OPEN_BASE + "development_manifest.json"), "development", config, root
    )
    dataset = ControlledWorldTensorDataset(tensorized, WorldActionEncoder())
    counts = defaultdict(list)
    for record in dataset.records:
        counts[record["family_uid"]].append(int(record["target_mask"].sum()))
    if set(counts) != set(FAMILY_UIDS) or any(
        len(set(v)) != 1 for v in counts.values()
    ):
        raise ValueError(
            "cannot reconstruct pooled NLL from unequal within-family masks"
        )
    family_weights = {f: sum(v) for f, v in counts.items()}
    source_bytes(root, OPEN_BASE + "calibration_manifest.json", sources)
    calibration_tensorized = tensorize_manifest(
        root / (OPEN_BASE + "calibration_manifest.json"), "calibration", config, root
    )
    calibration_dataset = ControlledWorldTensorDataset(
        calibration_tensorized, WorldActionEncoder()
    )
    methods = {r["method_id"]: r for r in protocol["methods"]}
    ledger_rows = {r["assignment_uid"]: r for r in ledger["assignments"]}
    rows = []
    all_pairs = []
    support_count = 0
    for assignment in manifest["assignments"]:
        base = assignment["output_root"]
        method = methods[assignment["method_id"]]
        status = read_json(root, base + "/terminal_status.json", sources)
        nofit = not method["checkpoint_creation_allowed"]
        curve = (
            []
            if nofit
            else read_rows(root, base + "/metrics/dev_metrics.jsonl", sources)
        )
        worlds = read_rows(
            root, base + "/evaluation/development_world_results.jsonl", sources
        )
        if any(
            w["assignment_uid"] != assignment["assignment_uid"]
            or w["seed"] != assignment["seed"]
            or w["role"] != "development"
            or w["test_outcome_access"] != 0
            for w in worlds
        ):
            raise ValueError("world evaluation identity/role mismatch")
        independent = independent_pair_metrics(
            worlds, tensorized["manifest"]["counterfactual_pairs"]
        )
        retained = read_rows(
            root, base + "/evaluation/development_pair_results.jsonl", sources
        )
        by_uid = {p["counterfactual_pair_uid"]: p for p in retained}
        if len(by_uid) != len(retained) or set(by_uid) != {
            p["counterfactual_pair_uid"] for p in independent
        }:
            raise ValueError("retained pair denominator mismatch")
        for p in independent:
            for metric in LOWER_IS_BETTER_METRICS:
                a, b = p[metric], by_uid[p["counterfactual_pair_uid"]][metric]
                if (a is None) != (b is None) or (
                    a is not None
                    and not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
                ):
                    raise ValueError("independent retained pair metric mismatch")
        all_pairs.extend(independent)
        support = read_rows(
            root, base + "/evaluation/support_abstention_results.jsonl", sources
        )
        if not support or any(
            r["status_match"] is not True or r["abstention_compliant"] is not True
            for r in support
        ):
            raise ValueError("support/abstention evidence invalid")
        support_count += len(support)
        row = diagnose_assignment(
            assignment, method, status, curve, family_weights, policy
        )
        row["checkpoint_identity"], row["identity_blockers"] = checkpoint_identity(
            root,
            assignment,
            ledger_rows[assignment["assignment_uid"]],
            curve,
            worlds,
            dataset,
            sources,
            policy,
        )
        if not nofit:
            run_config = yaml.safe_load(
                source_bytes(root, base + "/config.resolved.yaml", sources)
            )
            if run_config["operational_twin_training_protocol"] != protocol:
                row["identity_blockers"].append("frozen_run_protocol_drift")
            cal = read_rows(root, base + "/metrics/calibration_metrics.jsonl", sources)
            if (
                len(cal) != 1
                or cal[0]["calibration_eligible"] is not method["calibration_eligible"]
            ):
                row["identity_blockers"].append("calibration_contract_mismatch")
            row["calibration_summary"] = cal[0] if len(cal) == 1 else None
            if method["calibration_eligible"]:
                cal_worlds = read_rows(
                    root, base + "/evaluation/calibration_world_results.jsonl", sources
                )
                checkpoint = torch.load(
                    root
                    / ledger_rows[assignment["assignment_uid"]]["checkpoint"]["path"],
                    map_location="cpu",
                    weights_only=True,
                )
                cal_replay = replay_checkpoint(
                    checkpoint,
                    calibration_dataset,
                    assignment,
                    cal_worlds,
                    policy["checkpoint_replay"],
                    role="calibration",
                )
                row["checkpoint_identity"]["calibration_replay"] = cal_replay
                if cal_replay["archived_metric_mismatches"]:
                    row["identity_blockers"].append(
                        "archived_calibration_metrics_not_reproduced"
                    )
        else:
            nofit_rows = read_rows(
                root,
                base + "/" + assignment["runtime_contract"]["metrics_path"],
                sources,
            )
            row["calibration_summary"] = None
            if not nofit_rows:
                row["identity_blockers"].append("no_fit_metrics_missing")
        rows.append(row)
    method_rows = aggregate_methods(rows, manifest["method_ids"], manifest["seeds"])
    statistics_rows = independent_statistics(
        all_pairs, manifest["seeds"], candidate["statistical_plan"]
    )
    # Reuse production helper as a second comparison, never as the independent calculation.
    production = build_development_comparisons(all_pairs, candidate["statistical_plan"])
    retained_stats = candidate["development_paired_statistics"]["comparisons"]
    for group in (production["comparisons"], retained_stats):
        lookup = {(r["baseline_method_id"], r["metric"]): r for r in group}
        for independent in statistics_rows:
            expected = lookup[(independent["baseline"], independent["metric"])]
            if independent["status"] != expected["status"]:
                raise ValueError("undefined statistics disagreement")
            if independent["status"] == "UNDEFINED":
                continue
            for key in ("estimate", "lower", "upper"):
                if not math.isclose(
                    independent[key],
                    expected["family_stratified_bootstrap_ci95"][key],
                    rel_tol=1e-10,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "independent bootstrap replay disagreement: " + key
                    )
            if (
                independent["holm_adjusted_p_value"]
                != expected["holm_adjusted_p_value"]
            ):
                raise ValueError("independent Holm/permutation replay disagreement")
    summaries = []
    for method in manifest["method_ids"]:
        for family in FAMILY_UIDS:
            for seed in manifest["seeds"]:
                group = [
                    p
                    for p in all_pairs
                    if (p["method_id"], p["family_uid"], p["seed"])
                    == (method, family, seed)
                ]
                summaries.append(
                    {
                        "method_id": method,
                        "family_uid": family,
                        "seed": seed,
                        "world_count": len(group),
                        "metrics": {
                            m: statistics.fmean(p[m] for p in group if p[m] is not None)
                            if any(p[m] is not None for p in group)
                            else None
                            for m in LOWER_IS_BETTER_METRICS
                        },
                    }
                )
    cpu06 = []
    for kind, receipts in ledger["receipt_groups"].items():
        for receipt in receipts:
            scheduler = receipt.get("scheduler", {})
            if scheduler.get("node") == "cpu06":
                cpu06.append(
                    {
                        "receipt_uid": receipt["receipt_uid"],
                        "kind": kind,
                        "job_id": scheduler["job_id"],
                        "source": receipt["source"]["path"],
                        "state": scheduler["state"],
                        "exit_code": scheduler["exit_code"],
                    }
                )
    historical_policy = []
    for commit in sorted(
        set(candidate["training_code_commits"] + candidate["evaluation_code_commits"])
    ):
        payload = subprocess.check_output(
            ["git", "show", commit + ":AGENTS.md"], cwd=root
        )
        text = payload.decode()
        historical_policy.append(
            {
                "commit": commit,
                "path": "AGENTS.md",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "explicit_cpu04_cpu03_language_present": "CPU tests should use `cpu03` or `cpu04`"
                in text,
            }
        )
    # No receipt grants itself historical permission. Node mismatch is not a scientific identity failure.
    reconciliation = {
        "status": "UNKNOWN_BUT_NON_MATERIAL",
        "receipts": cpu06,
        "historical_policy": historical_policy,
        "authorization_metadata": "no explicit cpu06 exception found in bounded ledger and Git-bound policy",
        "scientific_identity_affected_by_node_name_alone": False,
        "boundary": "historical authorization remains unresolved; current audit runs only on allowed cpu04",
    }
    counts = Counter(r["classification"] for r in rows)
    identity_rows = [r for r in rows if r["identity_blockers"]]
    if identity_rows:
        terminal = "OPERATIONAL_TWIN_TRAINING_EVIDENCE_INVALID"
    elif counts[STATES[3]]:
        terminal = "OPERATIONAL_TWIN_NUMERICALLY_INVALID"
    elif counts[STATES[2]]:
        terminal = "OPERATIONAL_TWIN_NONCONVERGENT"
    elif counts[STATES[1]]:
        terminal = "OPERATIONAL_TWIN_TRAINING_INSUFFICIENT"
    elif counts[STATES[4]]:
        terminal = "OPERATIONAL_TWIN_TRAINING_SUFFICIENCY_INDETERMINATE"
    else:
        terminal = "SUFFICIENCY_AWAITING_INDEPENDENT_REVIEW"
    common = {
        "schema_version": SCHEMA_VERSION,
        "evidence_class": "controlled_world_non_empirical",
        "policy_sha256": sources[POLICY_PATH]["sha256"],
        "audit_execution_sha256": hashlib.sha256(
            json.dumps(execution, sort_keys=True).encode()
        ).hexdigest(),
        "boundary": dict(CLOSED),
    }
    decision = {
        **common,
        "artifact_type": "decision",
        "terminal": terminal,
        "assignment_count": len(rows),
        "classification_counts": dict(counts),
        "identity_invalid_assignment_count": len(identity_rows),
        "independent_review_status": "PENDING_NOT_SELF_APPROVED",
        "scientific_twin_result": "NOT_RUN",
        "continuation_authorized": False,
        "presealed_freeze_created": False,
        "next_action": "independent_review_of_checkpoint_selection_integrity_and_optimization_evidence",
        "irreducible_boundary": "Do not replace the frozen best-dev identity with final weights, relabel checkpoints, retrain, or open sealed evidence.",
    }
    analysis = {
        **common,
        "artifact_type": "mixed_metrics",
        "independent_pair_metric_replay": "PASS",
        "independent_statistical_replay": "PASS",
        "statistics_policy": candidate["statistical_plan"],
        "comparisons": statistics_rows,
        "method_family_seed_metrics": summaries,
        "denominators": {
            "assignments": len(rows),
            "method_seed_world_executions": len(rows) * len(dataset),
            "method_seed_latent_world_pairs": len(all_pairs),
            "independent_latent_worlds": len(
                tensorized["manifest"]["counterfactual_pairs"]
            ),
            "seeds": manifest["seeds"],
            "support_cases": support_count,
        },
        "root_cause_assessment": [
            {
                "hypothesis": "undertraining",
                "status": "PER_SEED_DIAGNOSTICS_REQUIRED",
                "evidence": "see optimization classifications; budget exhaustion is not proof; checkpoint defects are separate",
            },
            {
                "hypothesis": "objective_mismatch",
                "status": "SUPPORTED_DESIGN_DIFFERENCE",
                "evidence": "Full UQ trains Gaussian NLL but frozen checkpoint selection minimizes RMSE; no change to either rule is authorized",
            },
            {
                "hypothesis": "decision_vs_trajectory_tradeoff",
                "status": "SUPPORTED_DESCRIPTIVE_NOT_CAUSAL",
                "evidence": "report all eight primary comparisons, including regret/direction gains, action-contrast Holm nonsignificance against same-backbone, and worse trajectory RMSE",
            },
            {
                "hypothesis": "distributional_vs_point_prediction_tradeoff",
                "status": "INDETERMINATE",
                "evidence": "different loss/capacity and selected-weight identity confound causal attribution",
            },
            {
                "hypothesis": "action_head_weakness",
                "status": "NOT_ESTABLISHED",
                "evidence": "action sensitivity exists; Holm-nonsignificant contrast is not proof of no effect",
            },
            {
                "hypothesis": "trajectory_misspecification",
                "status": "INDETERMINATE",
                "evidence": "worse RMSE cannot separate misspecification from objective and checkpoint selection defects",
            },
            {
                "hypothesis": "family_heterogeneity",
                "status": "DESCRIPTIVE_EVIDENCE_RETAINED",
                "evidence": "all 45 method-family-seed rows retained; no favorable family selection",
            },
            {
                "hypothesis": "seed_instability",
                "status": "DESCRIPTIVE_EVIDENCE_RETAINED",
                "evidence": "all three seeds retained in per-seed diagnostics and every comparison",
            },
            {
                "hypothesis": "calibration_problem",
                "status": "INDETERMINATE_FOR_LEARNING_DYNAMICS",
                "evidence": "one post-training calibration state per UQ seed; uncalibrated coverage trajectories and calibrated replay retained; no per-epoch refit",
            },
            {
                "hypothesis": "checkpoint_selection_aliasing",
                "status": "SUPPORTED" if identity_rows else "NOT_OBSERVED",
                "evidence": "per-assignment optimizer-step, retained-last tensor comparison and method-aware open-dev replay; not a scientific Twin FAIL",
            },
        ],
    }
    audit = {
        **common,
        "artifact_type": "audit",
        "execution_profile": execution,
        "historical_replay_evidence": history,
        "terminal": terminal,
        "assignment_count": len(rows),
        "checkpoint_count": sum(
            r["checkpoint_identity"]["checkpoint_created"] for r in rows
        ),
        "identity_invalid_assignment_count": len(identity_rows),
        "classification_counts": dict(counts),
        "ledger_status": ledger["status"],
        "historical_provenance_exception_count": ledger[
            "historical_provenance_exception_count"
        ],
        "retained_attempt_count": sum(
            len(a["attempts"]) for a in ledger["assignments"]
        ),
        "frozen_ledger_dependency_count": len(ledger["dependencies"]),
        "audit_only_config_stack_extension": True,
        "cpu06_reconciliation": reconciliation,
        "family_valid_target_element_counts": family_weights,
        "unopened_receipt": unopened,
        "source_files": sorted(sources.values(), key=lambda r: r["path"]),
        "original_candidate_mutated": False,
        "training_executed": False,
        "calibration_refit": False,
        "historical_full_suite": {
            "passed": 1082,
            "failed": 210,
            "errors": 19,
            "rerun_by_this_audit": False,
        },
    }
    return {
        "audit": audit,
        "per_seed": {**common, "artifact_type": "per_seed", "assignments": rows},
        "methods": {**common, "artifact_type": "methods", "methods": method_rows},
        "mixed_metrics": analysis,
        "decision": decision,
    }


def render_report(artifacts: Mapping) -> str:
    decision = artifacts["decision"]
    audit = artifacts["audit"]
    lines = [
        "# Operational Twin training sufficiency audit",
        "",
        f"Terminal: `{decision['terminal']}`.",
        "Controlled-world non-empirical evidence. Independent review is pending; sealed scientific qualification is NOT_RUN.",
        "",
        f"Assignments: {audit['assignment_count']}; checkpoints: {audit['checkpoint_count']}; invalid selected identities: {audit['identity_invalid_assignment_count']}.",
        f"Optimization classifications: `{json.dumps(audit['classification_counts'], sort_keys=True)}`.",
        "Checkpoint identity failures are not evidence of scientific Twin failure or numerical nonconvergence.",
        "",
        "| Method | Seed | Optimization | Labeled epoch | Replayed RMSE | Selected RMSE | Identity |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for r in artifacts["per_seed"]["assignments"]:
        ck = r["checkpoint_identity"]
        lines.append(
            f"| {r['method_id']} | {r['seed']} | {r['classification']} | {ck.get('labeled_epoch', 'n/a')} | {ck.get('replay', {}).get('reconstruction_rmse', 'n/a')} | {ck.get('selected_recorded_rmse', 'n/a')} | {ck['status']} |"
        )
    lines += [
        "",
        "## Mixed development evidence",
        "Independent pair metrics and bootstrap/permutation/Holm replay retain all 15 assignments, all three seeds and all families.",
        "Full UQ improves ranking regret and effect direction against same-backbone no-action, but action-contrast is Holm-nonsignificant and trajectory RMSE is worse. Persistence comparisons are separately retained; they must not replace the stronger baseline.",
        "NLL-versus-MSE objectives and RMSE selection differ. The checkpoint defect confounds causal attribution of utility tradeoffs. No post-hoc primary metric or threshold is introduced.",
        "",
        "## Evidence boundary and recoverability",
        "The labeled early best-dev checkpoint can contain later weights and optimizer tensors. No replacement selected-epoch identity was found among the explicitly declared retained best/last checkpoints. Originals, logs, candidate, hashes, protocol and unopened receipts are unchanged.",
        "Three no-fit assignments are included without manufactured checkpoints/UQ. Optional gradient-norm, intermediate-checkpoint and per-epoch calibration series are unavailable.",
        f"Historical provenance exceptions remain {audit['historical_provenance_exception_count']}; retained attempts {audit['retained_attempt_count']}. cpu06: UNKNOWN_BUT_NON_MATERIAL (authorization unresolved, node name alone does not invalidate scientific identity).",
        "Historical full suite remains 1082 passed / 210 failed / 19 errors, not repository-wide green.",
        "",
        "## CPU audit replay identity and retained observations",
        "The explicit single-thread CPU profile is chosen to remove parallel-reduction scheduling, not to suppress discrepancies. Original audit settings were not recorded and are not inferred from its four-CPU allocation.",
        f"Effective Torch intra-op/inter-op: {audit['execution_profile']['torch_intra_op_threads']}/{audit['execution_profile']['torch_inter_op_threads']}; deterministic algorithms enabled; native thread startup environment and software/build/CPU identity are bound in `audit.execution_profile`.",
        "Prior three discrepancy records remain in `audit.historical_replay_evidence.pre_profile_audit`, sourced from commit 417623382420a51ef3701d427471270d7f827e15. The independent one-thread observations (electrophysiology0005/0015, including the newly observed0005 discrepancy) and four-thread diagnostic comparison are preserved in the embedded immutable review from commit 2c6fb750a7cc79d37976310715d3b241040fc584, with path and SHA-256 provenance. Historical observations do not overwrite canonical replay or establish the original runtime profile.",
        "The original criterion freeze remains bound to its Git identity; only audit execution settings were added. Scientific replay tolerances and all selected-checkpoint defects are unchanged.",
        "",
        "## Reproduction",
        "Run on cpu04/cpu (or allowed cpu03), with set -euo pipefail and node-local TMPDIR. Stage explicit code, ledger dependencies, open manifests and the twelve checkpoint paths plus existing declared last.pt files. Do not stage or read sealed/test payloads.",
        "`PYTHONPATH=Libs:. python Libs/scripts/builders/build_operational_twin_training_sufficiency.py`",
        "`PYTHONPATH=Libs:. python Libs/scripts/validators/validate_operational_twin_training_sufficiency.py --read-only`",
        "Both CLIs normalize the declared native-library environment in a fresh interpreter and enforce the effective Torch profile before replay. Direct library calls with a mismatched startup environment fail closed. Exact verification also requires the recorded CPU/software/build identity; cross-hardware numerical equivalence is not asserted.",
        "The builder exits nonzero when the scientific readiness gate is not met, after writing the honest audit. The read-only validator checks artifact integrity/recomputation; it does not grant scientific approval.",
        "",
        "## Next action",
        decision["irreducible_boundary"],
        "Independent review must decide the reached evidence boundary; no continuation authority, presealed freeze, AgentTrace, Agent, Tool, Memory or Co-Evolution gate is opened.",
    ]
    return "\n".join(lines) + "\n"


NATIVE_ARTIFACT_TYPE = "manuscript_native_training_evidence"
NATIVE_CONFIG_PATH = "Libs/configs/exp/manuscript_native_protocol.yaml"
NATIVE_OUTPUT_NAME = "manuscript_native_training_evidence.json"
NATIVE_COMPONENT_ARTIFACT_TYPE = "native_predictor_training_evidence"
NATIVE_COMPONENT_OUTPUT_NAME = "native_predictor_training_evidence.json"
NATIVE_REVIEW_REGISTRY = "Libs/configs/validation/native_predictor_training_reviews.yaml"
NATIVE_REVIEW_ROOT = Path(__file__).resolve().parents[3]


def strict_json(path: Path) -> dict:
    """Reject nonstandard numbers and duplicate keys in authoritative evidence."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key: " + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("nonfinite JSON number: " + value)
    value = json.loads(path.read_text(), object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(value, dict) or not finite_tree(value):
        raise ValueError("evidence must be a finite JSON object")
    return value


def native_evidence_path(root: Path, reference: str, training_root: Path | None = None,
                         execution_root: str | None = None) -> Path:
    """Relocate only the declared execution locator; keep embedded identities intact."""
    path = Path(reference)
    if any(word in part.lower() for part in path.parts for word in
           ("protected", "sealed", "private", "six_context", "holdout")):
        raise ValueError("protected evidence path denied")
    if ".." in path.parts:
        raise ValueError("parent traversal denied")
    if training_root is not None and execution_root is not None and not path.is_absolute():
        try:
            path = Path(training_root) / path.relative_to(execution_root)
        except ValueError:
            path = root / path
    elif not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    if any(word in part.lower() for part in resolved.parts for word in
           ("protected", "sealed", "private", "six_context", "holdout")):
        raise ValueError("protected evidence target denied")
    return resolved


def replay_native_selection(selection: Mapping, policy: Mapping) -> dict:
    """Replay scalar selection only. This is not optimization or forecast replay."""
    from Libs.utils.model.native_training_selection import NativeTrainingSelection
    if not isinstance(selection, Mapping):
        raise ValueError("native selection must be a mapping")
    if selection.get("selection_role") != "development":
        raise ValueError("native selection role differs from development")
    history = selection.get("history")
    if not finite_tree(selection):
        raise ValueError("nonfinite native selection evidence")
    if not isinstance(history, list) or not history:
        raise ValueError("ordinary native selection history unavailable")
    replay = NativeTrainingSelection(dict(policy))
    for row in history:
        if replay.terminal != "RUNNING":
            raise ValueError("native history continues after terminal selection")
        if (not isinstance(row, dict) or type(row.get("epoch")) is not int
                or type(row.get("steps")) is not int or row["epoch"] < 1 or row["steps"] < 1
                or type(row.get("selected_best")) is not bool
                or any(type(row.get(key)) not in (int, float) for key in
                       ("training_loss", "development_loss", "elapsed_seconds"))
                or row["elapsed_seconds"] < 0):
            raise ValueError("malformed native selection snapshot")
        replay.update(role="development", epoch=row["epoch"], steps=row["steps"],
                      training_loss=row["training_loss"], development_loss=row["development_loss"],
                      checkpoint_path="scalar_snapshot", elapsed_seconds=row["elapsed_seconds"])
        if replay.history[-1] != row:
            raise ValueError("native selected-best history mismatch")
    expected = replay.payload()
    for key in ("best", "last"):
        recorded = selection.get(key)
        if not isinstance(recorded, dict) or not isinstance(recorded.get("checkpoint_path"), str):
            raise ValueError("native retained " + key + " checkpoint reference unavailable")
        if {k: v for k, v in recorded.items() if k != "checkpoint_path"} != {
                k: v for k, v in expected[key].items() if k != "checkpoint_path"}:
            raise ValueError("native selected " + key + " scalar identity mismatch")
    # A producer can terminate on a failed partial validation. Preserve that failure
    # as incomplete evidence rather than infer a completed snapshot.
    if selection.get("terminal") != expected["terminal"]:
        raise ValueError("native terminal selection mismatch")
    if selection.get("sufficient_training") is not False or selection.get("scientific_twin") != "NOT_RUN":
        raise ValueError("native selection attempts scientific promotion")
    return {"status": "EXACT_SCALAR_REPLAY", "terminal": expected["terminal"],
            "snapshots": len(history), "stale_snapshots": replay.stale_epochs,
            "best": expected["best"], "last": expected["last"]}


def native_fit_contract(config: Mapping) -> dict:
    """Actual source, input, model and optimization settings consumed by the fit.

    Later feedback/calibration policies and audit reporting are separate consumers.
    Compare complete fit configuration sections rather than deleting arbitrary keys
    from the full configuration hash. The executed full hash is checked separately.
    """
    benchmark = config["screening_pipeline"]["screening_benchmark"]
    sections = ("native_profile_identity", "native_scientific_training",
                "native_diffusion_data", "native_input_staging", "native_observable_bridge",
                "native_romani_source", "controlled_cns_source", "native_learned_distillation",
                "native_rssm_prediction", "neuronsifter_scale_selection",
                "selected_formal_seed", "source_policy", "foundation_twin")
    return {**{key: config[key] for key in sections},
            "model_architecture": benchmark["model_architecture"],
            "training": benchmark["training"],
            "artifact_root": benchmark["artifact_root"]}


def native_executed_profile(root: Path, manifest_path: Path, expected: Mapping,
                            specification: Mapping, bind, read_json) -> tuple[dict, dict] | None:
    """Consume a reviewed execution chain for a required method/fraction/seed row."""
    from Libs.utils.evaluation.manuscript_native_protocol import content_hash, training_profile
    chain_path = manifest_path.parent / "execution_chain.json"
    if not chain_path.is_file():
        return None
    chain = read_json(chain_path)
    identity = expected["native_profile_identity"]
    authority = expected["manuscript_native_protocol"]["training_evidence"].get(
        "approved_execution_profiles", {}).get(specification["profile_id"])
    if not isinstance(authority, Mapping):
        raise ValueError("required native execution profile has no independently reviewed authority")
    shared = chain.get("input_mode") == "completed_shared_stage"
    if shared:
        if set(authority) != {"binding_kind", "execution_chain", "component_review"} or authority["binding_kind"] != "native_source_comparison_execution.v1":
            raise ValueError("incomplete shared-stage native execution authority")
        if chain.get("schema_version") != authority["binding_kind"]:
            raise ValueError("shared-stage execution schema mismatch")
        review = read_json(native_evidence_path(root, authority["component_review"]))
        if (review.get("status") != "APPROVED_COMPONENT" or review.get("profile_id") != specification["profile_id"]
                or review.get("execution_chain_sha256") != hashlib.sha256(chain_path.read_bytes()).hexdigest()):
            raise ValueError("shared-stage execution lacks exact independent approval")
        hash_authority = review
        shared_stage = chain.get("shared_input_stage", {})
        data_path = native_evidence_path(root, shared_stage["execution_chain_reference"])
        if hashlib.sha256(bind(data_path).read_bytes()).hexdigest() != shared_stage.get("execution_chain_sha256"):
            raise ValueError("shared-stage frozen data chain hash mismatch")
        data_chain = read_json(data_path)
        if any(data_chain.get(key) != value for key, value in chain["admission"].items()):
            raise ValueError("shared-stage frozen source/admission identity mismatch")
        if shared_stage.get("stage") != "/tmp/NrnSift_native_training_inputs_" + str(shared_stage.get("job_id")):
            raise ValueError("shared-stage original CPU03 stage identity mismatch")
    else:
        if expected["screening_pipeline"]["screening_benchmark"]["training"].get("seed_policy") != "manuscript_native_five":
            raise ValueError("historical execution authority cannot supply active three-seed evidence")
        if set(authority) != {"execution_chain", "source_commit", "profile_sha256", "effective_configuration_sha256"}:
            raise ValueError("incomplete approved native execution profile authority")
        hash_authority = authority
    for key in ("source_commit", "profile_sha256", "effective_configuration_sha256"):
        value = hash_authority[key]
        width = 40 if key == "source_commit" else 64
        if not isinstance(value, str) or len(value) != width or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("invalid approved native execution authority: " + key)
        if chain.get(key) != value:
            raise ValueError("reviewed native execution authority mismatch: " + key)
    approved_chain = read_json(native_evidence_path(root, authority["execution_chain"]))
    if content_hash(approved_chain) != content_hash(chain):
        raise ValueError("durable execution chain differs from approved retained authority")
    if chain.get("profile_reference") != specification["config_path"] or chain.get("output_reference") != specification["output_root"]:
        raise ValueError("retained execution profile/output identity mismatch")
    package = native_evidence_path(root, chain["frozen_package"])
    retained_chain = read_json(package / "execution_chain.json")
    if content_hash(retained_chain) != content_hash(chain):
        raise ValueError("durable execution chain differs from retained frozen package")
    profile_name = chain["profile_file"]
    if not isinstance(profile_name, str) or Path(profile_name).name != profile_name:
        raise ValueError("retained profile filename must be a package basename")
    profile_path = bind(package / profile_name)
    profile_bytes = profile_path.read_bytes()
    if hashlib.sha256(profile_bytes).hexdigest() != chain["profile_sha256"]:
        raise ValueError("retained execution profile byte hash mismatch")
    actual = yaml.safe_load(profile_bytes)
    if not isinstance(actual, Mapping) or not finite_tree(actual) or content_hash(actual) != chain["effective_configuration_sha256"]:
        raise ValueError("retained execution profile content hash mismatch")
    for reference in ("Libs/utils/evaluation/manuscript_native_protocol.py",
                      "Libs/utils/model/native_scientific_source.py",
                      "Libs/exp/training/pretrain_microenvironment_world_model.py"):
        source = subprocess.check_output(["git", "show", chain["source_commit"] + ":" + reference], cwd=root)
        if hashlib.sha256(source).hexdigest() != chain["code_sha256"].get(reference):
            raise ValueError("retained execution source hash mismatch: " + reference)
    if native_fit_contract(actual) != native_fit_contract(expected):
        raise ValueError("retained execution scientific contract mismatch")
    _, actual_spec = training_profile(actual, identity["method"], identity["seed"],
                                      training_component=identity.get("training_component"))
    for key in ("config_path", "output_root", "training_manifest", "training_seed_binding",
                "budget", "checkpoint_scale", "inference_profile", "source_scope"):
        if actual_spec[key] != specification[key]:
            raise ValueError("retained execution required profile mismatch: " + key)
    if Counter(actual_spec["family_scope"]) != Counter(specification["family_scope"]):
        raise ValueError("retained execution required profile mismatch: family_scope")
    evidence = {"path": str(profile_path), "effective_configuration_sha256": content_hash(actual),
                "execution_chain": str(chain_path), "source_commit": chain["source_commit"],
                "profile_sha256": chain["profile_sha256"], "scientific_contract": "MATCH"}
    return dict(actual), evidence


def build_native_audit(root: Path, config: Mapping, *, config_path: str = NATIVE_CONFIG_PATH,
                       training_root: Path | None = None, training_component: str | None = None) -> dict:
    """Inspect existing native metadata without producing sources, models or fits."""
    from Libs.utils.evaluation.manuscript_native_protocol import (
        content_hash, method_roster, training_profile, training_component_definition,
    )
    from Libs.utils.model.native_scientific_source import (
        native_training_profile, prior_particle_count, selected_group_definitions,
    )
    from Libs.utils.exp.native_comparator_training import comparator_identity
    from Libs.utils.model.native_training_selection import training_checkpoint_path

    root = Path(root).resolve()
    protocol = config["manuscript_native_protocol"]
    native_training_seed_values(protocol.get("seed_policy", "manuscript_native_five"), protocol["training_seeds"])
    if prior_particle_count(config) != 32:
        raise ValueError("native audit requires the complete 32-tuple prior")
    if config_path != protocol["config_path"]:
        raise ValueError("native audit requires the canonical protocol configuration")
    sources = {}
    cache = {}
    execution_root = protocol["execution_root"]
    checkpoint_limit = config["native_scientific_training"]["checkpoint_storage"]["maximum_payload_bytes"]
    checkpoints = {}

    def bind(path, *, checkpoint=False):
        path = native_evidence_path(root, str(path))
        key = str(path)
        if key in sources:
            return path
        size = path.stat().st_size
        limit = checkpoint_limit if checkpoint else 16 * 1024 * 1024
        if size > limit:
            raise ValueError("metadata/checkpoint exceeds bounded evidence read limit")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        sources[key] = {"bytes": size, "sha256": digest.hexdigest()}
        return path

    def read_json(path):
        return strict_json(bind(path))

    for reference in (config_path, SCHEMA_PATH,
                      "Libs/utils/evaluation/operational_twin_training_sufficiency.py",
                      "Libs/utils/evaluation/manuscript_native_protocol.py",
                      "Libs/utils/model/native_training_selection.py",
                      "Libs/utils/model/native_scientific_source.py",
                      "Libs/utils/exp/native_comparator_training.py",
                      "Libs/exp/training/pretrain_microenvironment_world_model.py",
                      "Libs/scripts/builders/build_operational_twin_training_sufficiency.py",
                      "Libs/scripts/validators/validate_operational_twin_training_sufficiency.py"):
        bind(root / reference)

    roster = method_roster(config)
    component = training_component_definition(config, training_component) if training_component else None
    if component:
        roster = [next(row for row in roster if row["method_id"] == component["conditioning_method"]),
                  {"method_id": component["predictor_method"], "software_interface_bound": True,
                   "implementation_scope": "internal_predictor_training_dependency"}]
    derived = {}
    for method in roster:
        if method["software_interface_bound"]:
            for seed in protocol["training_seeds"]:
                internal = training_component if component and method["method_id"] == component["predictor_method"] else None
                derived[(method["method_id"], seed)] = training_profile(
                    config, method["method_id"], seed, training_component=internal)
    rows = []
    for method in roster:
        method_id = method["method_id"]
        for seed in protocol["training_seeds"]:
            row = {"profile_id": f"{method_id}_fraction_100_seed_{seed}", "method": method_id,
                   "seed": seed, "training_subset_percentage": 100, "required": True,
                   "status": "MISSING", "issues": [], "expected_profile": None,
                   "profile_evidence": None, "training_evidence": None,
                   "checkpoint_evidence": [], "selection_replay": None, "receipts": {},
                   "visitation_proof": "UNAVAILABLE_NO_PER_GROUP_SEQUENCE",
                   "optimizer_transition_replay": "UNAVAILABLE_NOT_RECORDED",
                   "numerical_prediction_replay": "NOT_RUN_METADATA_ONLY",
                   "implementation_scope": method.get("implementation_scope"),
                   "later_gate_obligations": method.get("later_gate_obligations"),
                   "implementation_blocker": method.get("implementation_blocker")}
            rows.append(row)
            if not method["software_interface_bound"]:
                row["status"] = "EXTERNAL_OR_LATER_GATE_BLOCKED"
                row["issues"].append("required_implementation_or_external_evidence_unavailable")
                continue
            effective, spec = derived[(method_id, seed)]
            row["profile_id"] = spec["profile_id"]
            row["expected_profile"] = spec
            if method_id == "persistence" or spec["training_manifest"] is None:
                row["status"] = "NO_FIT_OR_DEPENDENT_CHECKPOINT"
                row["issues"].append("no_independent_fit_sufficiency_claim")
                continue
            owner = spec.get("checkpoint_reuse_method", method_id)
            owner_effective, owner_spec = derived[(owner, seed)]
            row["checkpoint_owner_profile"] = owner_spec["profile_id"]
            identity = (owner, seed)
            if identity not in cache:
                evidence = {"issues": [], "profile_evidence": None, "training_evidence": None,
                            "checkpoint_evidence": [], "selection_replay": None, "receipts": {}}
                cache[identity] = evidence
                issues = evidence["issues"]
                try:
                    manifest_path = native_evidence_path(root, owner_spec["training_manifest"],
                                                        training_root, execution_root)
                    executed = native_executed_profile(root, manifest_path, owner_effective,
                                                       owner_spec, bind, read_json)
                    if executed is not None:
                        owner_effective, evidence["profile_evidence"] = executed
                    else:
                        if component:
                            issues.append("approved_executed_profile_required")
                        profile_path = native_evidence_path(root, owner_spec["config_path"])
                        if not profile_path.is_file():
                            issues.append("actual_profile_missing")
                        else:
                            actual = yaml.safe_load(bind(profile_path).read_text())
                            evidence["profile_evidence"] = {"path": str(profile_path),
                                "effective_configuration_sha256": content_hash(actual)}
                            if content_hash(actual) != owner_spec["effective_configuration_sha256"]:
                                issues.append("actual_profile_identity_or_budget_mismatch")
                    manifest = read_json(manifest_path)
                    evidence["training_evidence"] = {"path": str(manifest_path), "manifest": manifest}
                    if manifest.get("training_seed_binding") != owner_spec["training_seed_binding"]:
                        issues.append("training_seed_binding_mismatch")
                    if manifest.get("failure") is not None:
                        issues.append("training_failure_recorded")
                    selection = manifest.get("selection", {})
                    if not isinstance(selection, Mapping):
                        raise ValueError("native selection must be a mapping")
                    try:
                        evidence["selection_replay"] = replay_native_selection(
                            selection, owner_effective["native_scientific_training"]["selection"])
                    except (ValueError, TypeError, KeyError, FloatingPointError) as exc:
                        issues.append("selection_replay_invalid: " + str(exc))
                    if selection.get("terminal") != "DEVELOPMENT_PLATEAU_REQUIRES_SUFFICIENCY_REVIEW":
                        issues.append("training_partial_failed_or_capped")
                    last_snapshot = selection.get("last")
                    if not isinstance(last_snapshot, Mapping):
                        issues.append("last_snapshot_missing_or_malformed")
                        last_snapshot = {}
                    if manifest.get("executed_steps") != last_snapshot.get("steps"):
                        issues.append("executed_steps_disagree_with_last_snapshot")
                    for role in ("train", "development"):
                        groups = selected_group_definitions(owner_effective, role)
                        counts = dict(Counter(g["family_uid"] for g in groups))
                        if manifest.get(role + "_group_count") != len(groups):
                            issues.append(role + "_group_count_mismatch")
                        if manifest.get(role + "_family_group_counts") != counts:
                            issues.append(role + "_family_group_counts_mismatch")
                    if manifest.get("scientific_twin", "NOT_RUN") != "NOT_RUN" or manifest.get("formal_qualification", False) is not False:
                        issues.append("manifest_scientific_promotion_rejected")
                    if type(manifest.get("executed_group_visits")) is not int or manifest.get("executed_group_visits", -1) < 0:
                        issues.append("aggregate_group_visits_unavailable")
                    native = owner in protocol["native_method_profiles"]
                    if native:
                        profile = manifest.get("native_scientific_profile")
                        if not isinstance(profile, dict) or not isinstance(profile.get("normalization_id"), str):
                            issues.append("native_scientific_profile_unavailable")
                        elif profile != native_training_profile(owner_effective, profile["normalization_id"]):
                            issues.append("native_scientific_profile_mismatch")
                        capacity = manifest.get("actual_total_parameter_count")
                        if manifest.get("checkpoint_scale") != owner_spec["checkpoint_scale"] or manifest.get("inference_profile") != owner_spec["inference_profile"]:
                            issues.append("full_capacity_metadata_unavailable_or_mismatched")
                    else:
                        recorded = manifest.get("identity", {})
                        if not isinstance(recorded, Mapping):
                            raise ValueError("comparator identity must be a mapping")
                        if not isinstance(recorded.get("conditioning", {}), Mapping):
                            raise ValueError("comparator conditioning must be a mapping")
                        if manifest.get("method") != owner or recorded != comparator_identity(
                                owner_effective, owner, recorded.get("conditioning", {})):
                            issues.append("comparator_profile_identity_mismatch")
                        capacity = manifest.get("actual_predictor_parameter_count")
                    if type(capacity) is not int or capacity < 1:
                        issues.append("actual_parameter_count_unavailable")
                    evidence["capacity_evidence"] = {"recorded_parameter_count": capacity,
                        "proof": "recorded_metadata_not_optimizer_or_prediction_replay"}
                    conditioning = manifest.get("frozen_conditioning") if native else manifest.get("identity", {}).get("conditioning")
                    if not owner_spec["joint_belief_fit"]:
                        if not isinstance(conditioning, dict) or conditioning.get("training_seed_binding") != owner_spec["training_seed_binding"] or conditioning.get("selection_role") != "development":
                            issues.append("frozen_conditioning_identity_unavailable_or_mismatched")
                        else:
                            for label, ref_key, hash_key in (("conditioning_config", "conditioning_config", "conditioning_config_sha256"),
                                                            ("conditioning_manifest", "conditioning_manifest", None)):
                                ref = native_evidence_path(root, owner_spec[ref_key], training_root, execution_root)
                                if not ref.is_file():
                                    issues.append(label + "_unavailable")
                                else:
                                    bind(ref)
                                    if hash_key and sources[str(ref)]["sha256"] != conditioning.get(hash_key):
                                        issues.append(label + "_hash_mismatch")
                            selected_ref = conditioning.get("checkpoint_path")
                            if isinstance(selected_ref, str):
                                cp = native_evidence_path(root, selected_ref, training_root, execution_root)
                                if cp.is_file():
                                    bind(cp, checkpoint=True)
                                    if sources[str(cp)]["sha256"] != conditioning.get("checkpoint_sha256"):
                                        issues.append("conditioning_checkpoint_hash_mismatch")
                                else:
                                    issues.append("conditioning_checkpoint_unavailable")
                            else:
                                issues.append("conditioning_checkpoint_reference_unavailable")
                    for name in ("best", "last"):
                        selected = selection.get(name)
                        if not isinstance(selected, dict) or not isinstance(selected.get("checkpoint_path"), str):
                            issues.append(name + "_checkpoint_reference_unavailable")
                            continue
                        checkpoint_path = training_checkpoint_path(manifest_path, selection, selected["checkpoint_path"])
                        if training_root is not None and selection.get("checkpoint_path_base", "repository") == "repository":
                            checkpoint_path = native_evidence_path(root, selected["checkpoint_path"], training_root, execution_root)
                        checkpoint_path = bind(checkpoint_path, checkpoint=True)
                        key = str(checkpoint_path)
                        payload_key = sources[key]["sha256"]
                        if payload_key not in checkpoints:
                            loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
                            if not isinstance(loaded, Mapping):
                                raise ValueError("native checkpoint payload must be a mapping")
                            checkpoints[payload_key] = {name: loaded.get(name) for name in
                                ("epoch", "executed_steps", "training_seed_binding", "native_scientific_profile", "native_comparator")}
                            checkpoints[payload_key]["state_available"] = (isinstance(loaded.get("model_state_dict"), dict)
                                and isinstance(loaded.get("optimizer_state_dict"), dict))
                            del loaded
                        checkpoint = checkpoints[payload_key]
                        entry = {"selection": name, "path": key, **sources[key],
                                 "epoch": checkpoint.get("epoch"), "executed_steps": checkpoint.get("executed_steps")}
                        evidence["checkpoint_evidence"].append(entry)
                        if checkpoint.get("epoch") != selected.get("epoch") or checkpoint.get("executed_steps") != selected.get("steps"):
                            issues.append(name + "_checkpoint_snapshot_mismatch")
                        if checkpoint.get("training_seed_binding") != owner_spec["training_seed_binding"]:
                            issues.append(name + "_checkpoint_seed_mismatch")
                        profile_key = "native_scientific_profile" if native else "native_comparator"
                        expected_identity = manifest.get("native_scientific_profile" if native else "identity")
                        if checkpoint.get(profile_key) != expected_identity:
                            issues.append(name + "_checkpoint_profile_mismatch")
                        if not checkpoint["state_available"]:
                            issues.append(name + "_checkpoint_state_unavailable")
                        del checkpoint
                    # Existing execution/resource receipts are optional observations,
                    # never inferred from a manifest's first-microbatch measurement.
                    for name in ("execution.json", "process_resources.log"):
                        receipt = manifest_path.parent / name
                        if receipt.is_file():
                            bind(receipt)
                            evidence["receipts"][name] = (read_json(receipt) if name.endswith(".json")
                                                         else {"path": str(receipt), **sources[str(receipt.resolve())]})
                        else:
                            issues.append(name + "_unavailable")
                except (OSError, ValueError, TypeError, KeyError, RuntimeError,
                        pickle.UnpicklingError, EOFError, yaml.YAMLError) as exc:
                    issues.append("missing_or_invalid_retained_evidence: " + str(exc))
            row.update(cache[identity])
            row["issues"] = list(row["issues"])
            if owner != method_id:
                try:
                    alias_path = native_evidence_path(root, spec["config_path"])
                    alias_profile = yaml.safe_load(bind(alias_path).read_text())
                    if content_hash(alias_profile) != spec["effective_configuration_sha256"]:
                        row["issues"].append("alias_profile_identity_or_budget_mismatch")
                except (OSError, ValueError, TypeError) as exc:
                    row["issues"].append("alias_profile_missing_or_invalid: " + str(exc))
            row["status"] = "INCOMPLETE_OR_INVALID" if row["issues"] else "EVIDENCE_REPLAYED_REVIEW_PENDING"
    # Read only the small admission verdict if available. Never open its source
    # arrays, cases, private worlds or protected outcomes.
    admissions = {}
    for reference in (str(Path(config["native_observable_bridge"]["output_root"]) / "native_consumption.json"),
                      str(Path(config["native_diffusion_data"]["registry_path"]).parent / "native_consumption.json"),
                      str(Path(config["native_diffusion_data"]["registry_path"]).parent / "native_consumption_recheck.json")):
        path = native_evidence_path(root, reference)
        try:
            admissions[reference] = {"status": "AVAILABLE", "receipt": read_json(path)}
        except (OSError, ValueError, TypeError) as exc:
            admissions[reference] = {"status": "UNAVAILABLE", "reason": str(exc)}
    distinct = {entry["sha256"] for row in rows for entry in row["checkpoint_evidence"]}
    artifact = {"schema_version": SCHEMA_VERSION, "artifact_type": NATIVE_ARTIFACT_TYPE,
            "protocol_scope": "manuscript-native", "protocol_uid": protocol["protocol_uid"],
            "evidence_class": "controlled_world_non_empirical", "boundary": dict(CLOSED),
            "sufficient_training": False, "ScientificTwin": "NOT_RUN", "independent_review": "PENDING",
            "terminal": "NATIVE_TRAINING_EVIDENCE_REQUIRES_INDEPENDENT_REVIEW",
            "configuration_sha256": content_hash(config), "selection_policy": dict(config["native_scientific_training"]["selection"]),
            "training_locator": str(Path(training_root).resolve()) if training_root is not None else None,
            "required_training_seeds": list(protocol["training_seeds"]), "required_fraction": 100,
            "complete_prior_tuples": 32, "required_row_count": len(rows), "rows": rows,
            "row_status_counts": dict(Counter(row["status"] for row in rows)),
            "distinct_retained_checkpoint_payloads": len(distinct), "admission_evidence": admissions,
            "source_evidence": sources, "excluded_fraction_evidence": [10, 25, 50],
            "limitations": ["Fraction 10/25/50 evidence is outside the required full-fraction matrix.",
                "Scalar selection replay does not establish sufficiently trained models.",
                "Per-group visitation sequence and optimizer transition replay are unavailable.",
                "Numerical forecasts, calibration, protected outcomes and biological evidence are not read.",
                "Recorded capacity metadata is checked against the declared profile; fitted tensor capacity is not independently reconstructed.",
                "First-microbatch resource measurements do not establish full-fit resource sufficiency.",
                "Historical 2 percent plateau classification is not applied to native method-specific losses."]}
    if component:
        component_issues = []
        try:
            source_admission = native_component_source_admission(root, config, admissions, sources)
        except (KeyError, TypeError, ValueError) as exc:
            source_admission = None
            component_issues.append("source_admission_unavailable_or_invalid: " + str(exc))
        if any(row["issues"] for row in rows):
            component_issues.append("six_row_training_prerequisites_incomplete")
        artifact.update(artifact_type=NATIVE_COMPONENT_ARTIFACT_TYPE,
            training_component=training_component, component_definition=component,
            component_issues=component_issues, source_admission=source_admission,
            prerequisite_status="INCOMPLETE" if component_issues else "SATISFIED_REVIEW_REQUIRED")
    return artifact


def native_component_source_admission(root: Path, config: Mapping, admissions: Mapping, sources: Mapping) -> dict:
    """Bind existing admission receipts without loading registry arrays or worlds."""
    base_ref = str(Path(config["native_observable_bridge"]["output_root"]) / "native_consumption.json")
    named_ref = str(Path(config["native_diffusion_data"]["registry_path"]).parent / "native_consumption.json")
    base, named = (admissions[ref]["receipt"] for ref in (base_ref, named_ref))
    required = config["native_observable_bridge"]["budget"]["cases"]
    if (base.get("status") != "complete" or named.get("status") != "complete"
            or named.get("operation") != "bind_retained_physical_candidates"
            or named.get("admitted_count") != required or base.get("admitted_count") != required
            or not named.get("source_identity") or named["source_identity"] != base.get("source_identity")
            or not named.get("array_values_unchanged") or not named.get("compiled_programs_unchanged")
            or native_evidence_path(root, named.get("original_consumption_ref", "")) != native_evidence_path(root, base_ref)):
        raise ValueError("component requires complete unchanged base and named source admissions")
    receipts = {}
    for name, ref in (("base_admission", base_ref), ("named_admission", named_ref)):
        path = str(native_evidence_path(root, ref))
        receipts[name] = {"path": path, **sources[path]}
    return {"scope": "admitted_controlled_source", "source_identity": named["source_identity"],
            "receipts": receipts}


def native_report(artifact: Mapping) -> str:
    title = "Native predictor training component evidence" if artifact["artifact_type"] == NATIVE_COMPONENT_ARTIFACT_TYPE else "Manuscript-native training evidence"
    return (f"# {title}\n\n"
            "Status: independent sufficiency review pending. ScientificTwin: NOT_RUN.\n"
            f"Required rows: {artifact['required_row_count']}; seeds: {artifact['required_training_seeds']}; fraction: 100.\n"
            f"Observed row states: {json.dumps(artifact['row_status_counts'], sort_keys=True)}.\n\n"
            + "\n".join("- " + item for item in artifact["limitations"]) + "\n")


def reviewed_native_predictor_training(config: Mapping, binding: Mapping,
                                       source_admission: Mapping) -> dict:
    """Consume repository-pinned independent judgement for one internal component.

    The canonical registry is outside audited configuration inputs. Registering
    a verdict therefore cannot change the audit the verdict already reviewed.
    A valid metadata replay remains a prerequisite, never scientific approval.
    """
    from jsonschema import Draft202012Validator
    from Libs.utils.data.raw_data_registry import load_config
    from Libs.utils.evaluation.manuscript_native_protocol import (
        content_hash, training_component_definition, training_profile,
    )
    from Libs.utils.evaluation.operational_twin_training_protocol import _git_commit_path_bytes
    from Libs.utils.data.native_input_staging import input_locations_from_config

    root = NATIVE_REVIEW_ROOT
    component_id = "residual_mpc_predictor"
    canonical = load_config(root / NATIVE_CONFIG_PATH)
    definition = training_component_definition(canonical, component_id)
    if definition["review_authority_config"] != NATIVE_REVIEW_REGISTRY:
        raise ValueError("native predictor review registry differs from canonical authority")
    schema = strict_json(root / SCHEMA_PATH)

    def validate(value, name):
        Draft202012Validator({"$ref": "#/$defs/" + name, "$defs": schema["$defs"]}).validate(value)

    def digest_bytes(path):
        path = native_evidence_path(root, str(path))
        maximum = canonical["native_scientific_training"]["checkpoint_storage"]["maximum_payload_bytes"]
        size = path.stat().st_size
        if size > maximum:
            raise ValueError("review evidence exceeds the declared checkpoint read bound")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return {"bytes": size, "sha256": digest.hexdigest()}

    def pinned_bytes(commit, reference):
        if Path(reference).is_absolute() or ".." in Path(reference).parts:
            raise ValueError("review authority requires a repository-relative Git path")
        path = native_evidence_path(root, reference)
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("review authority metadata exceeds read bound")
        actual = path.read_bytes()
        if actual != _git_commit_path_bytes(root, commit, reference):
            raise ValueError("native predictor review authority is uncommitted or changed")
        return path, actual

    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        _, registry_bytes = pinned_bytes(head, NATIVE_REVIEW_REGISTRY)
        registry = yaml.safe_load(registry_bytes)
        validate(registry, "native_predictor_review_registry")
        pin = registry["verdicts"].get(component_id)
        if pin is None:
            raise ValueError("canonical reviewed training sufficiency verdict is absent")
        subprocess.run(["git", "merge-base", "--is-ancestor", pin["commit"], head],
                       cwd=root, check=True, capture_output=True)
        verdict_path, verdict_bytes = pinned_bytes(pin["commit"], pin["path"])
        if hashlib.sha256(verdict_bytes).hexdigest() != pin["sha256"]:
            raise ValueError("native predictor review verdict hash mismatch")
        verdict = strict_json(verdict_path)
        validate(verdict, "native_predictor_training_verdict")
        if (verdict["decision"] != "APPROVED" or verdict["training_component"] != component_id
                or verdict["review_scope"] != definition["review_scope"]
                or set(verdict["findings"]) != set(definition["required_review_findings"])
                or any(item["status"] != "supported" for item in verdict["findings"].values())):
            raise ValueError("native predictor sufficiency review is pending, unsupported or wrong-scope")
        audit_path = native_evidence_path(root, verdict["audit"]["path"])
        if digest_bytes(audit_path)["sha256"] != verdict["audit"]["sha256"]:
            raise ValueError("native predictor training audit hash mismatch")
        audit = strict_json(audit_path)
        Draft202012Validator(schema).validate(audit)
        if (audit["artifact_type"] != NATIVE_COMPONENT_ARTIFACT_TYPE
                or audit["training_component"] != component_id or audit["component_definition"] != definition
                or audit["configuration_sha256"] != content_hash(canonical)
                or audit["prerequisite_status"] != "SATISFIED_REVIEW_REQUIRED" or audit["component_issues"]):
            raise ValueError("native predictor component audit is stale or prerequisites are incomplete")
        expected_rows = {(method, seed) for method in (definition["predictor_method"], definition["conditioning_method"])
                         for seed in (41, 42, 43)}
        rows = {(row["method"], row["seed"]): row for row in audit["rows"]}
        if set(rows) != expected_rows or len(audit["rows"]) != 6:
            raise ValueError("native predictor review requires all six full-fraction seed rows")
        for (method, seed), row in rows.items():
            internal = component_id if method == definition["predictor_method"] else None
            _, spec = training_profile(canonical, method, seed, training_component=internal)
            if (row["expected_profile"] != spec or row["issues"]
                    or row["status"] != "EVIDENCE_REPLAYED_REVIEW_PENDING"
                    or not isinstance(row["profile_evidence"], Mapping)
                    or row["profile_evidence"].get("scientific_contract") != "MATCH"
                    or not row["profile_evidence"].get("execution_chain")):
                raise ValueError("native predictor required executed profile prerequisites differ")
        for reference, identity in audit["source_evidence"].items():
            if digest_bytes(reference) != identity:
                raise ValueError("native predictor audit source evidence changed: " + reference)
        for finding in verdict["findings"].values():
            for reference in finding["evidence"]:
                if digest_bytes(reference["path"])["sha256"] != reference["sha256"]:
                    raise ValueError("independent predictor review finding evidence changed")
        audited_source = audit["source_admission"]
        if (source_admission.get("scope") != "admitted_controlled_source"
                or source_admission.get("source_identity") != audited_source["source_identity"]):
            raise ValueError("native predictor review source admission identity mismatch")
        locations = input_locations_from_config(config)
        for key, reference in audited_source["receipts"].items():
            path = source_admission[key]
            data = locations.read_text(path).encode() if locations else native_evidence_path(root, path).read_bytes()
            if hashlib.sha256(data).hexdigest() != reference["sha256"]:
                raise ValueError("native predictor runtime admission receipt changed")
        seed_binding = binding["model_identity"]["seed_binding"]
        seed = seed_binding["model_seed"]
        predictor = rows[(definition["predictor_method"], seed)]
        conditioning = rows[(definition["conditioning_method"], seed)]
        expected_config, spec = training_profile(canonical, definition["predictor_method"], seed,
                                                training_component=component_id)
        if seed_binding != spec["training_seed_binding"] or native_fit_contract(config) != native_fit_contract(expected_config):
            raise ValueError("native predictor runtime seed, source or fit profile changed")
        manifest = predictor["training_evidence"]["manifest"]
        normalizations = {rows[(definition["conditioning_method"], value)]["training_evidence"]["manifest"]
                          ["native_scientific_profile"]["normalization_id"] for value in (41, 42, 43)}
        if (binding["model_identity"] != manifest["identity"] or not binding.get("normalization_id")
                or normalizations != {binding["normalization_id"]}
                or binding["model_identity"]["conditioning"]["normalization_id"] != binding["normalization_id"]):
            raise ValueError("native predictor or conditioning/normalization identity changed")
        for row, identity, manifest_key in ((predictor, binding, "training_manifest"),
                (conditioning, binding["model_identity"]["conditioning"], "manifest_path")):
            best = [entry for entry in row["checkpoint_evidence"] if entry["selection"] == "best"]
            if len(best) != 1 or best[0]["sha256"] != identity["checkpoint_sha256"]:
                raise ValueError("native predictor reviewed selected checkpoint changed")
            reference = row["training_evidence"]["path"]
            if digest_bytes(identity[manifest_key]) != audit["source_evidence"][reference]:
                raise ValueError("native predictor reviewed training manifest changed")
    except (OSError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        raise ValueError("canonical reviewed training sufficiency authority unavailable: " + str(exc)) from exc
    # HEAD authenticates current authority above. Freeze only the registry bytes
    # and immutable verdict pin so unrelated commits preserve report reuse.
    return {"scope": "native_predictor_training_component", "training_component": component_id,
            "owner_method": definition["owner_method"], "scientific_checkpoint_sufficiency": True,
            "review": {**pin, "registry_path": NATIVE_REVIEW_REGISTRY,
                       "registry_sha256": hashlib.sha256(registry_bytes).hexdigest()},
            "audit": verdict["audit"], "selected_seed": seed,
            "selected_predictor_sha256": binding["checkpoint_sha256"],
            "selected_conditioning_sha256": binding["model_identity"]["conditioning"]["checkpoint_sha256"],
            "normalization_id": binding["normalization_id"], "boundary": dict(CLOSED)}
