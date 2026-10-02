"""Matched finite-law cost checks for Haar coupling representations.

These generated marginals are mathematical workloads, not biological evidence.
Run on a compute node. Each method gets a fresh process for independent peak RSS.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
from pathlib import Path
import resource
import time

import numpy as np
import scipy
import torch

from Libs.models.modules.neuronsifter_world_action import (
    GraphSpectralTemporalWaveletTokenizer,
)
from Libs.utils.evaluation.marginal_intervention_risk import temporal_synthesis_matrix
from Libs.utils.evaluation.wavelet_coupling_risk import (
    _dual_lower_bound,
    _transport_problem,
    clipped_haar_coupling_bounds,
)


def _case(value):
    try:
        time_count, levels, alphabet, bins = map(int, value.split(":"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Case must be time:levels:alphabet:bins."
        ) from exc
    if time_count < 2 or min(alphabet, bins) < 1 or levels < 1 or alphabet % bins:
        raise argparse.ArgumentTypeError(
            "Require at least two time points, positive levels and bins dividing alphabet."
        )
    return time_count, levels, alphabet, bins


def _worker(connection, case, seed, method, budget):
    time_count, levels, alphabet, bins = case
    torch.set_num_threads(1)
    code = torch.linspace(-0.7, 1.1, alphabet, dtype=torch.float64)
    codec = GraphSpectralTemporalWaveletTokenizer(4, 2, levels, 1.0, False)
    packed, metadata = codec.pack_temporal_wavelets(
        code.new_zeros((1, time_count, 1, 1))
    )
    generator = torch.Generator().manual_seed(seed)
    probabilities = torch.rand(
        packed.shape[1], alphabet, generator=generator, dtype=code.dtype
    )
    probabilities /= probabilities.sum(-1, keepdim=True)
    reference = torch.linspace(-0.4, 0.6, time_count, dtype=code.dtype)
    weights = 0.25 + torch.rand(time_count, generator=generator, dtype=code.dtype)
    if seed % 2:
        weights[1::2] *= -1
    offset = code.new_tensor(0.3)
    record = dict(
        time_count=time_count,
        levels=len(metadata["lengths"]),
        alphabet=alphabet,
        bins=bins,
        seed=seed,
        method=method,
        objective_kind="positive" if seed % 2 == 0 else "signed",
        alternative_loss=0.5 * time_count,
    )
    before = resource.getrusage(resource.RUSAGE_SELF)
    start = time.perf_counter()
    try:
        if method == "amplitude":
            result = clipped_haar_coupling_bounds(
                probabilities,
                code,
                metadata,
                reference,
                weights,
                offset,
                lower_bound=0.0,
                bin_count=bins,
                max_variables=budget,
            )
            lower, upper = (
                float(result["coupling_lower"]),
                float(result["coupling_upper"]),
            )
            entries = int(result["transport_variables"])
        else:
            synthesis = temporal_synthesis_matrix(codec, time_count, code)
            matrix, rhs, low_cost, high_cost, slices, _ = _transport_problem(
                probabilities.numpy(),
                code.numpy(),
                synthesis.numpy(),
                reference.numpy(),
                weights.numpy(),
                0.0,
                tuple(range(0, alphabet + 1, alphabet // bins)),
                budget,
            )
            lower, _, _ = _dual_lower_bound(low_cost, matrix, rhs, slices)
            negative_upper, _, _ = _dual_lower_bound(-high_cost, matrix, rhs, slices)
            lower, upper = lower + float(offset), -negative_upper + float(offset)
            entries = len(low_cost)
        record.update(
            status="passed",
            lower=lower,
            upper=upper,
            width=upper - lower,
            table_entries=entries,
            decision_resolved=(
                upper < record["alternative_loss"] or lower > record["alternative_loss"]
            ),
        )
    except (ValueError, ArithmeticError) as exc:
        record.update(
            status="budget_exceeded" if "budget" in str(exc).lower() else "failed",
            error=str(exc),
        )
    after = resource.getrusage(resource.RUSAGE_SELF)
    record.update(
        wall_seconds=time.perf_counter() - start,
        cpu_seconds=after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
        process_peak_rss_mib=after.ru_maxrss / 1024,
        cpu_affinity_count=len(os.sched_getaffinity(0)),
    )
    connection.send(record)
    connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=_case, action="append", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--max-variables", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_variables < 1:
        parser.error("--max-variables must be positive.")
    context = multiprocessing.get_context("spawn")
    records, comparisons = [], []
    for case in args.case:
        for seed in args.seeds:
            pair = {}
            methods = (
                ("ancestral", "amplitude")
                if seed % 2 == 0
                else ("amplitude", "ancestral")
            )
            for method in methods:
                receiver, sender = context.Pipe(duplex=False)
                process = context.Process(
                    target=_worker,
                    args=(sender, case, seed, method, args.max_variables),
                )
                process.start()
                sender.close()
                try:
                    record = receiver.recv()
                except EOFError:
                    record = dict(
                        status="failed",
                        case=case,
                        seed=seed,
                        method=method,
                        error="worker exited without a result",
                    )
                finally:
                    receiver.close()
                    process.join()
                if process.exitcode != 0:
                    record.update(status="failed", worker_exitcode=process.exitcode)
                records.append(record)
                pair[method] = record
                print(json.dumps(record), flush=True)
            if all(record["status"] == "passed" for record in pair.values()):
                a, b = pair["ancestral"], pair["amplitude"]
                error = max(abs(a[key] - b[key]) for key in ("lower", "upper"))
                scale = 1 + max(abs(a[key]) for key in ("lower", "upper"))
                comparisons.append(
                    dict(
                        case=case,
                        seed=seed,
                        endpoint_error=error,
                        endpoints_match=error <= 1e-7 * scale,
                        entry_ratio=a["table_entries"] / b["table_entries"],
                        wall_ratio=a["wall_seconds"] / b["wall_seconds"],
                        peak_rss_ratio=a["process_peak_rss_mib"]
                        / b["process_peak_rss_mib"],
                    )
                )
    success = (
        all(item["status"] != "failed" for item in records)
        and all(item["endpoints_match"] for item in comparisons)
        and all(math.isfinite(item["endpoint_error"]) for item in comparisons)
    )
    report = dict(
        status="passed_with_budget_exclusions" if success else "failed",
        scope="generated_finite_laws_not_biological_evidence",
        node=os.uname().nodename,
        job_id=os.environ.get("SLURM_JOB_ID"),
        max_variables=args.max_variables,
        versions=dict(
            torch=torch.__version__, scipy=scipy.__version__, numpy=np.__version__
        ),
        timing_scope="compilation_and_two_solves_excluding_imports_and_input_generation",
        memory_scope="independent_process_peak_RSS_including_imports_and_inputs",
        records=records,
        comparisons=comparisons,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
