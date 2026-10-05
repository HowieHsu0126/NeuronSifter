"""Execute the preregistered Round-23 prospective C2-S mechanism pool.

The pool uses the registered source-backed NEURON/RxD teacher and writes only
development artifacts. It never reads the canonical C2-S trajectory store or
its protected final-holdout rows.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Mapping

import neuron
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.exp.training.train_restricted_c2_vnext import (  # noqa: E402
    _json_sha256,
    _sha256_file,
    _structured_action_features,
)
from Libs.scripts.builders.build_restricted_c2_current_clamp_domain import (  # noqa: E402
    _source_archive,
)
from Libs.scripts.teacher_generation.run_neuron_rxd_microenvironment_teacher import (  # noqa: E402
    protocol_to_trajectory,
    run_protocol_batch,
    sanity_check,
)
from Libs.utils.data.raw_data_registry import load_config, write_json  # noqa: E402


def _validate_pool_contract(
    config: Mapping[str, Any], contract: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if contract.get("schema_version") != "neuronsifter-round23-prospective-mechanism-pool-v1":
        raise ValueError("Round-23 mechanism pool schema version drift")
    if contract.get("evidence_role") != "development_only":
        raise ValueError("Round-23 mechanism pool must remain development-only")
    if contract.get("publication_claim_eligible") is not False:
        raise ValueError("Round-23 mechanism pool cannot be publication-claim eligible")
    if contract.get("protected_test_access_allowed") is not False:
        raise ValueError("Round-23 mechanism pool opened protected-test access")
    protocols = [dict(row) for row in contract.get("protocols", [])]
    if len(protocols) < 4:
        raise ValueError("Round-23 mechanism pool requires at least four protocols")
    action_uids = [str(row.get("action_uid", "")) for row in protocols]
    pair_uids = [str(row.get("pair_uid", "")) for row in protocols]
    if any(not value for value in action_uids + pair_uids):
        raise ValueError("Round-23 mechanism protocols require action and pair identities")
    if len(action_uids) != len(set(action_uids)) or len(pair_uids) != len(set(pair_uids)):
        raise ValueError("Round-23 mechanism protocol identities must be unique")
    protected_uids = {str(value) for value in contract["protected_action_uids"]}
    if set(action_uids) & protected_uids:
        raise ValueError("Round-23 mechanism pool references a protected action identity")
    allowed_roles = {str(value) for value in contract["allowed_split_roles"]}
    if {str(row.get("split_role")) for row in protocols} - allowed_roles:
        raise ValueError("Round-23 mechanism pool contains an undeclared split role")
    if not any(row["split_role"] == "train" for row in protocols) or not any(
        row["split_role"] == contract["candidate_split_role"] for row in protocols
    ):
        raise ValueError("Round-23 mechanism pool requires calibration and candidate actions")
    target_charge = float(contract["design"]["absolute_current_charge_nA_ms"])
    protected = contract["protected_protocol_identity"]
    simulation_duration = float(
        config["dynamics_ready_registry"]["neuron_rxd_teacher"]["protocol_runtime"]
        ["simulation_duration_ms"]
    )
    for protocol in protocols:
        amplitude = float(protocol["amplitude_nA"])
        duration = float(protocol["duration_ms"])
        delay = float(protocol["delay_ms"])
        if not np.isfinite([amplitude, duration, delay]).all() or amplitude == 0.0:
            raise ValueError("Round-23 mechanism interventions require finite nonzero current")
        if duration <= 0.0 or delay < 0.0 or delay + duration > simulation_duration:
            raise ValueError("Round-23 mechanism protocol timing is outside the simulator window")
        if not np.isclose(abs(amplitude) * duration, target_charge, rtol=0.0, atol=1e-9):
            raise ValueError("Round-23 mechanism actions are not cost-matched by absolute charge")
        if (
            np.isclose(amplitude, float(protected["amplitude_nA"]))
            and np.isclose(duration, float(protected["duration_ms"]))
            and np.isclose(delay, float(protected["delay_ms"]))
        ):
            raise ValueError("Round-23 mechanism pool duplicates the protected protocol identity")
    return protocols


def _action_row(protocol: Mapping[str, Any], *, control: bool) -> dict[str, Any]:
    action_uid = (
        f"control_zero_for_{protocol['action_uid']}"
        if control
        else str(protocol["action_uid"])
    )
    return {
        "action_uid": action_uid,
        "action_role": "matched_control" if control else "intervention",
        "pair_uid": str(protocol["pair_uid"]),
        "dose_value": 0.0 if control else float(protocol["amplitude_nA"]),
        "dose_unit": "nA",
        "duration_ms": float(protocol["duration_ms"]),
        "delay_ms": float(protocol["delay_ms"]),
        "route": "intracellular_soma_current_clamp",
        "schedule": "single_pulse_after_resting_state",
        "target_set": ["soma_membrane"],
        "agent_set": ["no_external_agent" if control else "electrical_current"],
    }


def run(*, output_dir: Path, config_path: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Round-23 mechanism pool output already exists: {output_dir}")
    config = load_config(config_path)
    contract = config["round23_mechanism_pool"]
    protocols = _validate_pool_contract(config, contract)
    source_archive, source_hash = _source_archive(config)
    source_model_id = str(contract["source_model_id"])

    records: list[dict[str, Any]] = []
    runtimes: list[dict[str, float]] = []
    for protocol in protocols:
        for control in (False, True):
            action = _action_row(protocol, control=control)
            records.append(
                {
                    "action_uid": action["action_uid"],
                    "pair_uid": action["pair_uid"],
                    "split_role": str(protocol["split_role"]),
                    "action": action,
                }
            )
            runtimes.append(
                {
                    "stimulus_amplitude_nA": float(action["dose_value"]),
                    "stimulus_delay_ms": float(action["delay_ms"]),
                    "stimulus_duration_ms": float(action["duration_ms"]),
                }
            )

    payloads = run_protocol_batch(config, source_model_id, runtimes)
    trajectories: list[np.ndarray] = []
    for index, (record, payload) in enumerate(zip(records, payloads, strict=True)):
        sanity_check(config, payload)
        trajectory = protocol_to_trajectory(config, payload)
        if trajectory.size == 0 or not np.isfinite(trajectory).all():
            raise ValueError(
                f"Prospective mechanism action produced invalid trajectory: {record['action_uid']}"
            )
        record["simulation_index"] = index
        record["spike_count"] = float(payload["spike_count"])
        trajectories.append(trajectory)

    trajectory_array = np.stack(trajectories, axis=0).astype(np.float32)
    uncertainty_floor = float(
        config["dynamics_ready_registry"]["chembl_pbpk_parameterization"]
        ["uncertainty_floor"]
    )
    uncertainty = np.full(
        (len(records), trajectory_array.shape[-1]),
        uncertainty_floor,
        dtype=np.float32,
    )
    train_action_uids = {
        str(record["action_uid"])
        for record in records
        if record["split_role"] == "train"
    }
    feature_layout = config["screening_pipeline"]["screening_benchmark"][
        "model_architecture"
    ]["rollout_operator"]["regimen_feature_layout"]
    action_features = _structured_action_features(
        [record["action"] for record in records],
        feature_layout,
        train_action_uids=train_action_uids,
    )
    for record in records:
        pair = [row for row in records if row["pair_uid"] == record["pair_uid"]]
        if len(pair) != 2:
            raise ValueError("Round-23 prospective action does not have one matched control")
    output_dir.mkdir(parents=True, exist_ok=False)
    array_path = output_dir / "pool_arrays.npz"
    np.savez_compressed(
        array_path,
        trajectories=trajectory_array,
        uncertainties=uncertainty,
        action_features=action_features,
    )
    created_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": "neuronsifter-round23-prospective-mechanism-pool-result-v1",
        "created_at": created_at,
        "pool_id": contract["pool_id"],
        "status": "prospective_registered_tool_execution_complete",
        "config_path": config_path.resolve().relative_to(REPO_ROOT).as_posix(),
        "config_sha256": _sha256_file(config_path),
        "pool_contract_sha256": _json_sha256(contract),
        "candidate_split_role": contract["candidate_split_role"],
        "records": records,
        "arrays": {
            "path": array_path.name,
            "sha256": _sha256_file(array_path),
            "trajectory_shape": list(trajectory_array.shape),
            "uncertainty_shape": list(uncertainty.shape),
            "action_feature_shape": list(action_features.shape),
        },
        "registered_tool_execution": {
            "tool": contract["registered_tool"],
            "neuron_version": str(neuron.__version__),
            "source_model_id": source_model_id,
            "source_archive_path": source_archive.resolve().relative_to(REPO_ROOT).as_posix(),
            "source_archive_sha256": source_hash,
            "host": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "prospective_dispatch": True,
            "tool_outcome_count": len(records),
        },
        "protected_boundary": {
            "canonical_c2s_trajectory_store_accessed": False,
            "canonical_final_holdout_result_accessed": False,
            "protected_action_uids_referenced": False,
            "protected_test_access_count": 0,
        },
        "matched_control_complete": True,
        "finite_trajectory": bool(np.isfinite(trajectory_array).all()),
        "finite_uncertainty": bool(np.isfinite(uncertainty).all()),
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args()
    result = run(
        output_dir=Path(arguments.output_dir).resolve(),
        config_path=(REPO_ROOT / arguments.config).resolve(),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
