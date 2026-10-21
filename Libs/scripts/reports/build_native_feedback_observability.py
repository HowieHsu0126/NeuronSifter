"""Measure a bounded training-source feedback readout without source or model execution."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from dataclasses import asdict
from itertools import combinations
import json
from pathlib import Path
import resource
import shutil
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Libs.utils.data.native_input_staging import NativeInputLocations, file_digest
from Libs.utils.data.native_joint_source_storage import read_joint_source_reference
from Libs.utils.data.native_observable_profile import family_grid, observable_profile, quantity_support
from Libs.utils.data.physical_occupancy import digest
from Libs.utils.data.raw_data_registry import load_config
from Libs.models.modules.world_state import JointParticleSupport
from Libs.utils.model.native_arrived_observation import NativeCNSReadout, native_cns_readout, project_cns_readout
from Libs.utils.model.native_scientific_source import group_definitions


def statistics(values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("Diagnostic statistics require nonempty finite values")
    return {"count": int(values.size), "min": float(values.min()), "median": float(np.median(values)),
            "p90": float(np.quantile(values, .9)), "p95": float(np.quantile(values, .95)),
            "max": float(values.max())}


def difference_summary(delta, readout, serialization_bound=None) -> dict:
    delta = np.asarray(delta, dtype=np.float64)
    mask = np.asarray(readout.observed, dtype=bool)
    if delta.shape != mask.shape or not mask.any() or not np.isfinite(delta).all():
        raise ValueError("Readout difference must contain the declared finite coordinates")
    scaled = delta / readout.standard_deviation
    l2 = float(np.linalg.norm(scaled[mask]))
    result = {"difference_over_noise_by_coordinate": scaled.tolist(),
              "observed_l2_over_single_observation_noise": l2,
              "observed_l2_over_independent_difference_noise": l2 / np.sqrt(2.),
              "observed_rms_over_noise": float(np.sqrt(np.mean(scaled[mask]**2))),
              "observed_max_absolute_over_noise": float(np.abs(scaled[mask]).max())}
    if serialization_bound is not None:
        bound = np.asarray(serialization_bound, dtype=np.float64)
        if bound.shape != delta.shape or not np.isfinite(bound).all() or (bound < 0).any():
            raise ValueError("Invalid declared projection serialization bound")
        result.update(projected_float32_serialization_bound=bound.tolist(),
                      observed_coordinates_exceeding_serialization_bound=int((np.abs(delta[mask]) > bound[mask]).sum()))
    return result


def verify_initial_projection(initial, retained_means, readout) -> tuple[np.ndarray, dict]:
    """Account for the producer's float32 scaled coordinates and mixed means."""
    projected = project_cns_readout(torch.as_tensor(initial), readout).numpy()
    selected = np.stack([initial[..., r, c, q] for r, c, q in readout.selector], -1)
    scales, matrix = np.asarray(readout.scales), np.asarray(readout.matrix)
    scaled = selected.astype(np.float64) / scales
    # This is serialization arithmetic only, following native_scientific_source.draw_group.
    source_scaled = np.asarray([[np.float32(value) / scale for value, scale in zip(row, readout.scales)]
                                for row in selected], dtype=np.float32).astype(np.float64)
    mixed = source_scaled @ matrix.T
    source_means = mixed.astype(np.float32).astype(np.float64)
    bound = np.abs(source_scaled - scaled) @ np.abs(matrix).T + np.abs(source_means - mixed)
    operations = 2 * len(readout.selector)
    unit_roundoff = np.finfo(np.float64).eps / 2
    gamma = operations * unit_roundoff / (1 - operations * unit_roundoff)
    bound += 2 * gamma * (np.abs(scaled) @ np.abs(matrix).T)
    retained = np.asarray(retained_means, dtype=np.float64)
    error = np.abs(projected - retained)
    if retained.shape != projected.shape or not np.isfinite(retained).all() or (error > bound).any():
        raise ValueError("Initial projection differs beyond the producer serialization precision")
    return projected, {"status": "within_declared_serialization_precision",
        "initial_field_dtype": str(initial.dtype), "retained_mean_dtype": "float32",
        "arithmetic": "float32_scaled_coordinates_float64_mixing_float32_mean",
        "maximum_absolute_error": float(error.max()), "maximum_derived_rounding_bound": float(bound.max())}


def normalized_numerical_error(quality: dict) -> float:
    """Read either existing source receipt form without treating missing evidence as zero."""
    keys = [key for key in ("max_normalized_error", "normalized_max") if key in quality]
    if len(keys) != 1 or (keys[0] == "normalized_max" and quality.get("admissible") is not True):
        raise ValueError("Numerical receipt lacks one recognized admitted error summary")
    value = quality[keys[0]]
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value) or value < 0:
        raise ValueError("Numerical receipt requires a finite nonnegative error")
    return float(value)


def reference_serialization_error(quality: dict) -> np.ndarray:
    """The long-CNS producer records this same reference cast inside components."""
    normalized_numerical_error(quality)
    if "max_normalized_error" in quality:
        values = quality["serialization_error"]
    else:
        values = quality["components"]["reference_float32_serialization"]
    errors = np.asarray(values, dtype=np.float64)
    if errors.ndim != 1 or not errors.size or not np.isfinite(errors).all() or (errors < 0).any():
        raise ValueError("Reference serialization errors must be finite nonnegative per-species values")
    return errors


def validate_training_receipt(receipt: dict, index: int, expected_group: dict, identity: dict) -> None:
    group = receipt["group"]
    stripped = {key: value for key, value in group.items() if key != "conditioning_reference"}
    if receipt["group_index"] != index or stripped != expected_group or group["role"] != "train":
        raise ValueError("Training group/index/definition mismatch before payload access")
    if receipt["source_identity"] != identity:
        raise ValueError("Training receipt source identity differs")
    cases = receipt["cases"]
    if len(cases) != 3 or {row["action_index"] for row in cases} != {0, 1, 2}:
        raise ValueError("Diagnostic requires all three declared paired action cases")
    for case in cases:
        quality = case.get("numerical", {})
        if (case["group_index"] != index or case.get("status") != "complete"
                or quality.get("reasons") != []):
            raise ValueError("Training case is incomplete or numerically rejected")
        normalized_numerical_error(quality)
        reference = Path(case["raw_reference"])
        if reference.name != str(reference) or not reference.name.startswith("case_") or not reference.name.endswith("_reference.npz"):
            raise ValueError("Only declared raw reference trajectories may be read")
    if not receipt["operations"] or any(row["status"] != "complete" for row in receipt["operations"].values()):
        raise ValueError("Training source operation remains incomplete")
    if len(group["particle_uids"]) != 32 or len(set(group["particle_uids"])) != 32:
        raise ValueError("Training diagnostic requires all 32 distinct initial tuples")
    if not group.get("conditioning_reference"):
        raise ValueError("Training complete-group conditioning reference is missing")


def numerical_receipt_delta(root: Path, first: int, stop: int, identity: dict) -> dict:
    """Read only explicitly named new receipts, stopping at an unfinished boundary."""
    cases, operations, kinds, reasons = Counter(), Counter(), Counter(), Counter()
    errors, clocks, read_bytes = [], [], 0
    completed, examined = 0, 0
    stop_index, status = first, "complete"
    for index in range(first, stop):
        try:
            payload = (root / f"group_{index}_receipt.json").read_bytes()
        except FileNotFoundError:
            status = "missing_receipt"
            break
        read_bytes += len(payload)
        receipt = json.loads(payload)
        examined += 1
        if receipt["group_index"] != index or receipt["source_identity"] != identity:
            status = "receipt_identity_mismatch"
            break
        valid = bool(receipt["cases"]) and bool(receipt["operations"])
        for case in receipt["cases"]:
            cases[case["status"]] += 1
            quality = case.get("numerical", {})
            case_reasons = quality.get("reasons", [])
            reasons.update(case_reasons)
            try:
                value = normalized_numerical_error(quality)
            except ValueError:
                value = None
                valid = False
            if value is not None:
                errors.append(float(value))
            valid &= case["status"] == "complete" and not case_reasons and value is not None and np.isfinite(value)
        for operation in receipt["operations"].values():
            operations[operation["status"]] += 1
            kinds[operation["kind"]] += 1
            valid &= operation["status"] == "complete"
        clocks.append(float(receipt["source_clock_seconds"]))
        if not valid:
            status = "incomplete_or_rejected_receipt"
            break
        completed += 1
        stop_index = index + 1
    return {"status": status, "requested_first_group": first, "requested_stop_exclusive": stop,
        "stop_index_exclusive": stop_index, "examined_groups": examined, "completed_accepted_groups": completed,
        "case_status_counts": dict(cases), "operation_status_counts": dict(operations),
        "operation_kind_counts": dict(kinds), "numerical_rejection_reasons": dict(reasons),
        "maximum_normalized_error": max(errors) if errors else None,
        "bytes_read": read_bytes, "first_source_clock_seconds": clocks[0] if clocks else None,
        "last_source_clock_seconds": clocks[-1] if clocks else None, "array_reads": 0}


def stage_explicit_inputs(root: Path, scratch: Path, receipts: list[dict]) -> NativeInputLocations:
    scratch.mkdir()
    files, aliases = {}, {}
    paths = set()
    for receipt in receipts:
        group = receipt["group"]
        paths.add(root / f"group_{receipt['group_index']}_receipt.json")
        reference = Path(group["conditioning_reference"]["path"])
        if reference.parent != root or not reference.name.endswith("_conditioning.json.gz"):
            raise ValueError("Conditioning reference is outside the declared production root")
        paths.add(reference)
        paths.update(root / case["raw_reference"] for case in receipt["cases"])
    for original in sorted(paths):
        target = scratch / "files" / original.relative_to("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        value = {"size_bytes": target.stat().st_size, "sha256": file_digest(target)}
        files[str(original)] = value
        aliases[str(original)] = str(original)
    return NativeInputLocations(scratch, aliases, files, frozenset(),
                                {"train": [receipt["group"]["sample_uid"] for receipt in receipts]})


def measure_group(config: dict, receipt: dict, locations: NativeInputLocations, root: Path) -> dict:
    group = receipt["group"]
    readout = native_cns_readout(config, group)
    raw, bindings = read_joint_source_reference(group["conditioning_reference"], input_locations=locations)
    support = JointParticleSupport.from_payload(raw["support"])
    if (support.role != "train" or support.sample_uids != (group["sample_uid"],)
            or support.particle_uids != (tuple(group["particle_uids"]),)
            or support.source_uid != group["source_uid"] or support.source_snapshot_ref != group["source_snapshot_ref"]):
        raise ValueError("Retained initial support differs from the training group")
    initial = support.initial_field[0].numpy()
    parameters = support.readout_parameters["controlled_predose_selector"]
    if parameters["law"] != "gaussian" or parameters["units"] != readout.units:
        raise ValueError("Initial Gaussian readout definition differs")
    variance = parameters["variance"][0].numpy()
    if not np.array_equal(variance, np.full_like(variance, readout.standard_deviation**2)):
        raise ValueError("Retained initial noise differs from the scoped declared noise")
    if raw["observations"][0]["observed"] != [list(readout.observed)]:
        raise ValueError("Retained initial mask differs from the scoped readout")
    means, consistency = verify_initial_projection(initial, parameters["mean"][0].numpy(), readout)
    mask = np.asarray(readout.observed, dtype=bool)
    pair_distances = [np.linalg.norm((means[i] - means[j])[mask]) / readout.standard_deviation
                      for i, j in combinations(range(len(means)), 2)]
    weights = support.prior_weights[0].double().numpy()
    weights /= weights.sum()
    center = weights @ means
    spread = np.sqrt(weights @ ((means - center)**2)) / readout.standard_deviation
    grid = family_grid(observable_profile(config), group["family_uid"])
    trajectories, projected, bounds, dtypes = [], [], [], []
    cases = sorted(receipt["cases"], key=lambda row: row["action_index"])
    for case in cases:
        logical = root / case["raw_reference"]
        path = locations.file_path(logical)
        if file_digest(path) != locations.files[str(logical)]["sha256"]:
            raise ValueError("Staged raw trajectory bytes changed")
        with np.load(path, allow_pickle=False) as arrays:
            field = arrays["trajectory"]
        expected = (len(grid["query_time"]["values"]), len(grid["region_id"]), len(grid["compartment_id"]), len(grid["species_id"]))
        if field.shape != expected or not np.isfinite(field).all():
            raise ValueError("Raw reference trajectory grid or finiteness mismatch")
        trajectories.append(field)
        dtypes.append(str(field.dtype))
        projected.append(project_cns_readout(torch.from_numpy(field), readout).numpy())
        errors = reference_serialization_error(case["numerical"])
        if any(q >= len(errors) for _, _, q in readout.selector):
            raise ValueError("Reference serialization evidence omits an observed species")
        selected = np.asarray([errors[q] / scale for (_, _, q), scale in zip(readout.selector, readout.scales)])
        bounds.append(np.abs(np.asarray(readout.matrix)) @ selected)
    if any(not np.array_equal(trajectories[0][0], field[0]) for field in trajectories[1:]):
        raise ValueError("Matched action trajectories do not share the same initial hidden state")
    regimen_uids = raw["particle_programs"]["regimen_uids"]
    if len(regimen_uids) != 3:
        raise ValueError("Retained complete group does not contain three action programs")
    rows = []
    for query in range(1, len(grid["query_time"]["values"])):
        pairs = [{"action_indices": [i, j], "regimen_uids": [regimen_uids[i], regimen_uids[j]],
                  **difference_summary(projected[i][query] - projected[j][query], readout, bounds[i] + bounds[j])}
                 for i, j in combinations(range(3), 2)]
        changes = [{"action_index": i, "regimen_uid": regimen_uids[i],
                    **difference_summary(projected[i][query] - projected[i][0], readout, 2 * bounds[i])}
                   for i in range(3)]
        rows.append({"query_index": query, "source_time": grid["query_time"]["values"][query],
            "source_time_unit": grid["query_time"]["unit"], "time_hours": grid["time_hours"][query],
            "action_pairs": pairs, "change_from_t0": changes})
    return {"group_index": receipt["group_index"],
        "group": {key: value for key, value in group.items() if key not in {"conditioning_reference", "particle_uids"}},
        "readout": asdict(readout), "conditioning_bindings": bindings,
        "case_uids": [case["case_uid"] for case in cases], "raw_trajectory_dtypes": dtypes,
        "initial_projection_check": consistency,
        "initial_tuple_pair_distances_over_noise": statistics(pair_distances),
        "initial_tuple_standard_deviation_over_noise_by_coordinate": spread.tolist(),
        "initial_tuple_count": len(means), "dynamic_between_tuple_spread": "UNMEASURED",
        "queries": rows}


def aggregate(groups: list[dict]) -> dict:
    by_stratum = defaultdict(list)
    for row in groups:
        by_stratum[row["group"].get("source_stratum", row["group"]["source_variant"])].append(row)
    result = {}
    for stratum, selected in by_stratum.items():
        pairs = [pair for group in selected for query in group["queries"] for pair in query["action_pairs"]]
        changes = [row for group in selected for query in group["queries"] for row in query["change_from_t0"]]
        result[stratum] = {"groups": len(selected),
            "action_pair_l2_over_noise": statistics([row["observed_l2_over_single_observation_noise"] for row in pairs]),
            "change_from_t0_l2_over_noise": statistics([row["observed_l2_over_single_observation_noise"] for row in changes]),
            "initial_tuple_pair_median_over_noise": statistics([row["initial_tuple_pair_distances_over_noise"]["median"] for row in selected]),
            "action_pair_observed_coordinate_count_above_serialization_bound": sum(row["observed_coordinates_exceeding_serialization_bound"] for row in pairs),
            "action_pair_observed_coordinate_denominator": sum(sum(group["readout"]["observed"]) * len(group["queries"]) * 3 for group in selected),
            "raw_trajectory_dtypes": sorted({value for group in selected for value in group["raw_trajectory_dtypes"]})}
    return result



def training_precision_contrasts(fields: list[np.ndarray], support: np.ndarray, readout: NativeCNSReadout) -> dict:
    """Measure the frozen source builder's float32 target cast, before any fit."""
    readout.validate()
    if (len(fields) != 3 or support.dtype != np.bool_ or support.ndim != 4
            or support.shape[0] < 2 or any(field.shape != support.shape or not np.isfinite(field).all() for field in fields)
            or any(not np.array_equal(fields[0][0], field[0]) for field in fields[1:])):
        raise ValueError("Precision diagnosis requires three finite aligned actions with shared t0 and boolean support")
    if any((field[~support] != 0).any() for field in fields):
        raise ValueError("Unsupported quantities must remain zero storage placeholders")
    # Same cast/order as the frozen source admission builder; no normalizer exists yet.
    stored = [field.astype(np.float32) for field in fields]
    if any(not np.isfinite(field).all() for field in stored):
        raise ValueError("The training-target float32 cast is nonfinite")
    raw_means = [project_cns_readout(torch.from_numpy(field), readout).numpy() for field in fields]
    stored_means = [project_cns_readout(torch.from_numpy(field), readout).numpy() for field in stored]
    raw_delta = np.stack([fields[a][1:].astype(np.float64)-fields[b][1:].astype(np.float64)
                          for a, b in combinations(range(3), 2)])
    stored_delta = np.stack([stored[a][1:].astype(np.float64)-stored[b][1:].astype(np.float64)
                             for a, b in combinations(range(3), 2)])
    active = np.broadcast_to(support[None, 1:], raw_delta.shape)
    def counts(before, after, valid):
        return {"coordinate_denominator": int(valid.sum()),
                "raw_nonzero": int(((before != 0) & valid).sum()),
                "stored_nonzero": int(((after != 0) & valid).sum()),
                "erased_nonzero": int(((before != 0) & (after == 0) & valid).sum()),
                "raw_max_abs": float(np.abs(before[valid]).max()) if valid.any() else None,
                "stored_max_abs": float(np.abs(after[valid]).max()) if valid.any() else None}
    quantities = {name: counts(raw_delta[..., index], stored_delta[..., index], active[..., index])
                  for index, name in enumerate(readout.species_ids)}
    projected_raw = np.stack([raw_means[a][1:]-raw_means[b][1:] for a, b in combinations(range(3), 2)])
    projected_stored = np.stack([stored_means[a][1:]-stored_means[b][1:] for a, b in combinations(range(3), 2)])
    observed = np.broadcast_to(np.asarray(readout.observed)[None, None], projected_raw.shape)
    readout_result = counts(projected_raw, projected_stored, observed)
    for label, delta in [("raw", projected_raw), ("stored", projected_stored)]:
        distance = np.linalg.norm(delta[..., np.asarray(readout.observed, dtype=bool)] / readout.standard_deviation, axis=-1)
        readout_result[label + "_l2_over_single_observation_noise"] = statistics(distance.reshape(-1))
    return {"reference_dtypes": [str(field.dtype) for field in fields], "training_target_dtype": "float32",
            "postzero_queries": support.shape[0]-1, "action_pairs": [[0, 1], [0, 2], [1, 2]],
            "by_quantity": quantities, "projected_observed_cns": readout_result}


def measure_retained_training_precision(report_path: Path, input_root: Path, config: dict, output: Path) -> dict:
    """Use only a previously verified TRAIN-only input package; no source replay."""
    original = json.loads(report_path.read_text())
    identity = original["source_identity"]
    if (original["status"] != "complete" or original["group_range"] != [0, 32]
            or len(original["groups"]) != 32 or identity["definition_sha256"] != digest({
                "profile": config["native_observable_bridge"], "background": config["controlled_cns_source"]})):
        raise ValueError("Retained training diagnostic/source definition mismatch")
    config = copy.deepcopy(config)
    config["native_arrived_observation_filter"] = copy.deepcopy(original["arrived_filter_policy"])
    files = original["staged_input_files"]
    locations = NativeInputLocations(input_root, {name: name for name in files}, files, frozenset(),
                                    {"train": [row["group"]["sample_uid"] for row in original["groups"]]})
    def verified(logical):
        path = locations.file_path(logical)
        if file_digest(path) != files[str(logical)]["sha256"]:
            raise ValueError("Retained training diagnostic input changed")
        return path
    result = {"schema_version": "native_feedback_observability.v1", "status": "started",
        "scope": "retained_training_target_cast_diagnostic", "source_identity": identity,
        "prior_report": str(report_path), "prior_report_sha256": file_digest(report_path),
        "source_generation_calls": 0, "fitted_normalizer_calls": 0, "model_calls": 0, "raw_array_reads": 0,
        "scientific_admission": "NOT_RUN", "normalization_and_quantization": "NOT_EVALUATED",
        "between_tuple_dynamic_identifiability": "UNMEASURED", "groups": [],
        "interpretation": "Exact source-target cast before final source admission; projected readouts from training targets are not acquired measurements."}
    logical_roots = {Path(name).parent for name in files if Path(name).name.startswith("group_") and name.endswith("_receipt.json")}
    if len(logical_roots) != 1:
        raise ValueError("Retained inputs require one source root")
    root = next(iter(logical_roots))
    started = time.perf_counter()
    try:
        for prior in original["groups"]:
            index, group = prior["group_index"], prior["group"]
            receipt = json.loads(verified(root / f"group_{index}_receipt.json").read_text())
            if (receipt["source_identity"] != identity or group["role"] != "train"
                    or any(receipt["group"].get(key) != value for key, value in group.items())):
                raise ValueError("Retained receipt differs from original TRAIN diagnostic")
            expected = {key: value for key, value in receipt["group"].items() if key != "conditioning_reference"}
            validate_training_receipt(receipt, index, expected, identity)
            cases = sorted(receipt["cases"], key=lambda row: row["action_index"])
            if [case["case_uid"] for case in cases] != prior["case_uids"]:
                raise ValueError("Paired action identities changed")
            readout = native_cns_readout(config, group)
            if json.loads(json.dumps(asdict(readout))) != prior["readout"]:
                raise ValueError("Retained readout differs from the frozen source law")
            fields = []
            for case in cases:
                with np.load(verified(root / case["raw_reference"]), allow_pickle=False) as data:
                    fields.append(data["trajectory"])
                    result["raw_array_reads"] += 1
            grid = family_grid(observable_profile(config), group["family_uid"])
            support = quantity_support(grid, group["source_uid"])
            result["groups"].append({"group_index": index, "group": group,
                **training_precision_contrasts(fields, support, readout)})
        result["status"] = "complete"
    except BaseException as error:
        result.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        result["elapsed_seconds"] = time.perf_counter()-started
        result["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--readout-config", type=Path, required=True)
    parser.add_argument("--expected-producer-commit", required=True)
    parser.add_argument("--first-group", type=int, required=True)
    parser.add_argument("--stop-group", type=int, required=True)
    parser.add_argument("--receipt-first", type=int, required=True)
    parser.add_argument("--receipt-stop", type=int, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.first_group < args.stop_group or not 0 <= args.receipt_first < args.receipt_stop:
        raise ValueError("Explicit nonempty group ranges are required")
    args.output.mkdir()
    started = time.perf_counter()
    report = {"schema_version": "native_feedback_observability.v1", "status": "started",
        "scope": "completed_training_source_diagnostic", "groups": [],
        "dynamic_between_tuple_spread": "UNMEASURED", "scientific_admission": "NOT_RUN"}
    try:
        root = args.source_root.resolve()
        config = load_config(args.source_config)
        definition_hash = digest({"profile": config["native_observable_bridge"], "background": config["controlled_cns_source"]})
        expected = group_definitions(config)[args.first_group:args.stop_group]
        receipts = [json.loads((root / f"group_{index}_receipt.json").read_bytes())
                    for index in range(args.first_group, args.stop_group)]
        identity = receipts[0]["source_identity"]
        if identity["producer_commit"] != args.expected_producer_commit or identity["definition_sha256"] != definition_hash:
            raise ValueError("Production identity differs from the declared frozen source")
        for index, group, receipt in zip(range(args.first_group, args.stop_group), expected, receipts):
            validate_training_receipt(receipt, index, group, identity)
        if len(expected) != len(receipts):
            raise ValueError("Requested group range exceeds the frozen source definition")
        current = load_config(args.readout_config)
        config = copy.deepcopy(config)
        config["native_arrived_observation_filter"] = copy.deepcopy(current["native_arrived_observation_filter"])
        delta = numerical_receipt_delta(root, args.receipt_first, args.receipt_stop, identity)
        (args.output / "new_numerical_receipts.json").write_text(json.dumps(delta, indent=2) + "\n")
        locations = stage_explicit_inputs(root, args.scratch, receipts)
        report.update(source_identity=identity, group_range=[args.first_group, args.stop_group],
            source_stratum_counts=dict(Counter(row["group"].get("source_stratum") for row in receipts)),
            source_config_sha256=file_digest(args.source_config), readout_config_sha256=file_digest(args.readout_config),
            arrived_filter_policy=config["native_arrived_observation_filter"],
            staged_input_files=locations.files, staged_input_bytes=sum(row["size_bytes"] for row in locations.files.values()),
            code_sha256={str(path.relative_to(ROOT)): file_digest(path) for path in (
                Path(__file__), ROOT/"Libs/utils/model/native_arrived_observation.py",
                ROOT/"Libs/utils/model/native_scientific_source.py", ROOT/"Libs/utils/data/native_joint_source_storage.py")},
            definitions={"noise_ratio": "Euclidean distance on observed mixed/scaled coordinates divided by declared sigma",
                "difference_noise_ratio": "same distance divided by sqrt(2)*sigma for two independent noisy acquisitions",
                "serialization_bound": "sum of per-case per-species float32 serialization maxima projected with absolute mixing weights",
                "prior_pairs": "all 496 unordered pairs among 32 retained INITIAL tuples; no evolved tuple fields exist",
                "raw_arrays": "reference trajectories kept in stored dtype; no downcast or fitted transform"})
        print(json.dumps({"inputs_staged": True, "groups": len(receipts), "files": len(locations.files),
                          "bytes": report["staged_input_bytes"]}), flush=True)
        for receipt in receipts:
            report["groups"].append(measure_group(config, receipt, locations, root))
        report["aggregate_by_source_stratum"] = aggregate(report["groups"])
        report["status"] = "complete"
        report["new_receipt_summary_status"] = delta["status"]
    except BaseException as error:
        report.update(status="failed", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (args.output / "observability.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "groups": len(report["groups"]),
                      "aggregate_by_source_stratum": report["aggregate_by_source_stratum"]}, indent=2))


if __name__ == "__main__":
    main()
