"""Validate Romani binding; explicitly opt into bounded real trajectory diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import tempfile
import time

import numpy as np
import yaml

from Libs.utils.data.physical_occupancy import digest
from Libs.utils.model import native_scientific_source as scientific

from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.model.native_romani_source import RomaniCell, definition, parameter_candidates, stage_source


def validate_binding(config: dict, source_root: Path, scratch: Path) -> dict:
    started = time.monotonic()
    policy = definition(config)
    prepared = stage_source(config, source_root, scratch)
    settings = config["native_observable_bridge"]
    basis = config["controlled_cns_source"]["factor_basis"]
    factors = {name: np.asarray([[settings["coefficient_levels"][(atom[fi]+coordinate) % 2]
                                 for coordinate in range(len(basis[name]))]
                                for atom in settings["atoms"]], dtype=float)
               for fi, name in enumerate(("z_tx", "z_pk", "z_env", "z_ephys", "z_morph"))}
    candidates = parameter_candidates(config, factors, np.random.default_rng(settings["seed"]))
    cell = RomaniCell(config, prepared)
    result = cell.verify_candidates(candidates)
    current = policy["common_current"]
    result["current_program_readbacks"] = [
        cell.configure_current(current["base_amplitude_uA"]*row["intensity_multiplier"],
                               row["delay_ms"], row["duration_ms"])
        for row in current["action_variants"]]
    # Compare only the exact files already copied by the maintained staging path.
    unchanged = all(hashlib.sha256((source_root / relative).read_bytes()).hexdigest() == expected
                    for relative, expected in prepared["files"].items())
    if not unchanged or result["final_time_ms"] != 0.:
        raise ValueError("Source immutability or time-zero binding check failed")
    result.update(source_files_unchanged=unchanged, source_files_compared=len(prepared["files"]),
                  source_preparation=prepared, status="PASS_PARAMETER_BINDING_ONLY",
                  elapsed_seconds=time.monotonic()-started,
                  maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  job_id=os.environ.get("SLURM_JOB_ID"), host=os.uname().nodename)
    return result


def validate_trajectories(config: dict, source_root: Path, scratch: Path, output: Path) -> dict:
    """Numerically inspect one declared prior, without admitting training data."""
    started = time.monotonic()
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_root = output.parent / (output.stem + "_raw")
    raw_root.mkdir(exist_ok=False)
    result = {"schema_version": "native_romani_trajectory_diagnostic.v1",
        "status": "started", "cases": [], "source_admission": "NOT_RUN",
        "training_records_admitted": 0, "model_calls": 0, "optimizer_updates": 0,
        "scientific_twin": "NOT_RUN", "formal_qualification": False,
        "forecast_parameter_selection": "all32_complete_prior_tuples_no_hidden_selector",
        "scope": "bounded_source_numerical_diagnostic_not_full_corpus_or_biological_validation",
        "numeric_interpretation": "fixed_step_differences_are_estimates_not_rigorous_error_bounds",
        "job_id": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename,
        "physical_trajectory_count": 0, "time_advance_calls": 0,
        "cns_integration_count": 0, "cns_rhs_evaluations": 0}
    def save():
        result["elapsed_seconds"] = time.monotonic()-started
        output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    def serializable(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError("Unexpected source metadata type: "+type(value).__name__)
    save()
    try:
        # Freeze actual loaded values before inspecting any trajectory. The global
        # production freeze gate is deliberately not entered by this diagnostic.
        snapshot = raw_root/"config_snapshot.yaml"
        snapshot.write_text(yaml.safe_dump(config, sort_keys=False))
        result["config_snapshot"] = str(snapshot)
        repo = Path(__file__).resolve().parents[3]
        source_paths = config["native_observable_bridge"]["producer_freeze_paths"]
        result["source_file_sha256"] = {name: hashlib.sha256((repo/name).read_bytes()).hexdigest()
                                        for name in source_paths}
        result["validator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        groups = scientific.group_definitions(config)
        group_index, group = next((index, row) for index, row in enumerate(groups)
                                 if row.get("source_law") == "romani_vm_source")
        if group["role"] != "train":
            raise ValueError("Romani diagnostic must use its declared training-source prior")
        scoped = scientific.scoped_group_config(config, group)
        policy = definition(scoped)
        result.update(group_definition=group, group_index=group_index,
                      source_definition=policy,
                      numerical_criteria=scoped["native_observable_bridge"]["numerics"])
        prepared = stage_source(scoped, source_root, scratch)
        result["source_preparation"] = prepared
        cell = RomaniCell(scoped, prepared)
        generated = scientific.draw_group(scoped, group, group_index,
            physical_initializer=lambda candidates: [cell.initialize(row) for row in candidates])
        support, parameters = generated["support"], generated["parameters"]
        actions = [scientific.action_definition(scoped, group["family_uid"], index, group)
                   for index in range(scientific.action_count(scoped, group))]
        if len(parameters) != 32 or len(actions) != 3:
            raise ValueError("Diagnostic must retain the original32 tuples and3 common actions")
        result.update(candidate_count=len(parameters), actions=actions,
                      planned_cases=len(parameters)*len(actions),
                      planned_physical_trajectories=3*len(parameters)*len(actions))
        producer = {key: generated[key] for key in
                    ("hidden_index", "parameters", "rng_streams", "parameter_readbacks")}
        (raw_root/"producer_truth.json").write_text(json.dumps(producer, indent=2, allow_nan=False))
        code_hash = digest(result["source_file_sha256"])
        programs = [[scientific.compiled_program(scoped, group, support, atom, values, action, code_hash)
                     for action in actions] for atom, values in enumerate(parameters)]
        conditioning = scientific.joint_source(scoped, group, generated, programs)
        input_path = raw_root/"initial_observations_and_programs.json"
        input_path.write_text(json.dumps(conditioning, allow_nan=False, default=serializable))
        result["observation_input"] = str(input_path)
        result["physical_truth_directory"] = str(raw_root)
        result["compiled_program_count"] = len(parameters)*len(actions)
        result["input_truth_boundary"] = "initial_observations_and_programs_exclude_future_arrays_and_hidden_selector"
        phases = ("coarse", "fine", "reference")
        methods = ("RK4", "RK4", "DOP853")
        fractions = scoped["native_observable_bridge"]["numerics"]["cns_max_step_interval_fractions"]
        result["cases"] = [{"particle_index": atom, "action_index": action,
                            "status": "planned", "phases": {}}
                           for atom in range(len(parameters)) for action in range(len(actions))]
        reference_voltage = {}
        for row in result["cases"]:
            atom, action_index = row["particle_index"], row["action_index"]
            row["status"] = "started"
            physical, coupled = {}, {}
            save()
            for phase, method, fraction in zip(phases, methods, fractions):
                physical[phase] = scientific.physical_trajectory(scoped, group["family_uid"],
                    parameters[atom], actions[action_index], refinement=phase, source_runtime=cell)
                result["physical_trajectory_count"] += 1
                result["time_advance_calls"] += physical[phase]["time_advance_calls"]
                name = f"particle_{atom:02d}_action_{action_index}_{phase}"
                physical_path = raw_root/(name+"_physical.json")
                physical_path.write_text(json.dumps(physical[phase], allow_nan=False, default=serializable))
                row["phases"][phase] = {"physical_path": str(physical_path),
                    "dt_ms": physical[phase]["dt_ms"],
                    "time_advance_calls": physical[phase]["time_advance_calls"]}
                save()
                law = scientific.CoupledObservableLaw(scoped, group, support, atom, physical[phase])
                coupled[phase] = scientific.integrate_coupled(law, method, fraction)
                result["cns_integration_count"] += 1
                result["cns_rhs_evaluations"] += coupled[phase]["nfev"]
                cns_path = raw_root/(name+"_cns.npz")
                np.savez_compressed(cns_path, **coupled[phase])
                row["phases"][phase].update(cns_path=str(cns_path),
                    cns_rhs_evaluations=coupled[phase]["nfev"])
            precise = dict(physical["reference"],
                fine_values=physical["fine"]["values"],
                precision_method="independent_fixed_step_refinements")
            quality = scientific.numerical_summary(scoped, group["family_uid"], parameters[atom],
                                                  physical["coarse"], precise, coupled)
            row.update(status="passed" if not quality["reasons"] else "failed_numerical_gate",
                       numerical=quality,
                       reference_voltage_range_mV=[float(precise["values"].min()),float(precise["values"].max())])
            reference_voltage[(atom, action_index)] = precise["values"].copy()
            save()
            print("ROMANI_TRAJECTORY_CASE", json.dumps({
                "particle": atom, "action": action_index, "status": row["status"],
                "max_normalized_error": quality["max_normalized_error"],
                "physical_trajectory_count": result["physical_trajectory_count"]}), flush=True)
        unchanged = all(hashlib.sha256((source_root/relative).read_bytes()).hexdigest() == expected
                        for relative, expected in prepared["files"].items())
        result.update(source_files_unchanged=unchanged, source_files_compared=len(prepared["files"]),
            passed_cases=sum(row["status"] == "passed" for row in result["cases"]),
            failed_cases=sum(row["status"] == "failed_numerical_gate" for row in result["cases"]),
            reference_action_differences_mV=[{
                "particle_index": atom, "action_pair": [left,right],
                "maximum_absolute_difference": float(np.abs(
                    reference_voltage[(atom,left)]-reference_voltage[(atom,right)]).max())}
                for atom in range(len(parameters)) for left,right in ((0,1),(0,2),(1,2))])
        if not unchanged:
            raise ValueError("Original ModelDB input files changed during the source diagnostic")
        result["status"] = ("PASS_NUMERICAL_DIAGNOSTIC" if result["passed_cases"] == result["planned_cases"]
                            else "FAILED_NUMERICAL_DIAGNOSTIC")
    except BaseException as error:
        result.update(status="FAILED_DIAGNOSTIC_EXECUTION",
                      failure={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        result["maxrss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        save()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="Libs/configs/exp/manuscript_native_protocol.yaml")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--scratch-parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectory-check", action="store_true",
                        help="Inspect all32 prior tuples/3 programs/3 fixed steps; never admit training data")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Parameter-binding evidence already exists: " + str(args.output))
    config = load_config(args.config)
    args.scratch_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="romani_binding_", dir=args.scratch_parent) as work:
        if args.trajectory_check:
            result = validate_trajectories(config, args.source_root.resolve(strict=True), Path(work), args.output)
            print(json.dumps({key: result[key] for key in
                ("status", "candidate_count", "passed_cases", "failed_cases",
                 "physical_trajectory_count", "time_advance_calls", "elapsed_seconds")}))
            return 0 if result["status"] == "PASS_NUMERICAL_DIAGNOSTIC" else 2
        result = validate_binding(config, args.source_root.resolve(strict=True), Path(work))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        print(json.dumps({key:result[key] for key in
              ("status","candidate_count","candidate_initializations","hoc_startup_initializations",
               "time_advance_calls","final_time_ms","trajectory_count","source_files_unchanged","elapsed_seconds")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
