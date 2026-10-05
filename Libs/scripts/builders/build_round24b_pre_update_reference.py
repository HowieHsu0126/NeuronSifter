"""Freeze the exact Round-24B posterior-bound pre-update reference."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.models.modules.round24b_posterior_adapter import (  # noqa: E402
    PosteriorBoundState,
    posterior_condition_probe,
)
from Libs.scripts.experiments.run_round23_shared_belief_replay import (  # noqa: E402
    _action_design_matrix,
    _fit_prior,
    _load_candidate_data,
    _load_world_action_outputs,
    _response_values,
)
from Libs.utils.data.raw_data_registry import load_config, write_json  # noqa: E402


DEFAULT_CONTRACT = REPO_ROOT / "Libs/configs/neuronsifter/round24b_posterior_bound_evaluation.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "Output/audits/round24b_pre_update_reference.json"
DEFAULT_POSTERIOR = REPO_ROOT / "Output/audits/round24b_pre_update_posterior.json"
DEFAULT_STATIC_FREEZE = REPO_ROOT / "Output/audits/round24b_static_reference_freeze.json"
DEFAULT_MECHANISM_RESOLUTION = REPO_ROOT / "Output/audits/round24b_round23_mechanism_resolution.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def load_contract(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "neuronsifter-round24b-posterior-bound-evaluation-v1":
        raise ValueError("Round-24B contract schema mismatch")
    boundary = payload["evidence_boundary"]
    if boundary != {
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
        "protected_holdout_access": 0,
        "external_comparison_open": False,
        "c2b_status": "blocked_insufficient_real_e2_e4",
    }:
        raise ValueError("Round-24B evidence boundary drift")
    if payload["updates"]["agent_parameters_update_allowed"] is not False:
        raise ValueError("Round-24B must freeze Agent parameters")
    return payload


def recompute_prior(
    contract: dict[str, Any], *, device: str
) -> tuple[PosteriorBoundState, dict[str, Any]]:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    reference = contract["reference"]
    static_path = repo_path(reference["static_reference_path"])
    static = json.loads(static_path.read_text(encoding="utf-8"))
    if static.get("status") != "frozen" or int(static.get("protected_final_holdout_access", -1)) != 0:
        raise ValueError("Round-24 static reference is not admissible")
    pool_path = repo_path(reference["c2s_pool_path"])
    pool_payload = json.loads(pool_path.read_text(encoding="utf-8"))
    data = _load_candidate_data(
        config,
        candidate_pool=str(pool_path),
        candidate_pool_identity=str(pool_path),
    )
    checkpoint_path = repo_path(reference["c2s_checkpoint_path"])
    truth, prediction, prediction_std, checkpoint_metadata = _load_world_action_outputs(
        config,
        data,
        checkpoint_path=checkpoint_path,
        checkpoint_identity_path=checkpoint_path,
        device=device,
    )
    if checkpoint_metadata["checkpoint_sha256"] != static["posterior_model_checkpoint"]["sha256"]:
        raise ValueError("Round-24B C2-S checkpoint differs from the frozen reference")
    species_ids = [str(row["species_id"]) for row in config["dynamics_ready_registry"]["species"]]
    response_truth, response_prediction, _, action_uids, _ = _response_values(
        data["records"], truth, prediction, prediction_std, species_ids
    )
    split_by_action = {
        str(row["action"]["action_uid"]): str(row["split_role"])
        for row in data["records"]
        if str(row["action"]["action_role"]) == "intervention"
    }
    train_uids = {uid for uid, split in split_by_action.items() if split == "train"}
    designs, _, _ = _action_design_matrix(data["records"], action_uids, train_uids)
    train_positions = [index for index, uid in enumerate(action_uids) if uid in train_uids]
    if len(train_positions) != 6:
        raise ValueError("Round-24B prior requires the six frozen train action pairs")
    old_truth = response_truth[train_positions]
    old_prediction = response_prediction[train_positions]
    old_designs = designs[train_positions]
    noise_scale = max(float(np.std(old_truth)) * 0.25, 1e-3)
    noise_variance = noise_scale**2
    prior = _fit_prior(old_designs, old_prediction, noise_variance)
    source_uid = str(static["scientific_posterior_uid"])
    state = PosteriorBoundState.from_gaussian(
        prior,
        source_posterior_uid=source_uid,
        source_checkpoint_sha256=checkpoint_metadata["checkpoint_sha256"],
        source_pool_manifest_sha256=sha256_file(pool_path),
        noise_variance=noise_variance,
        event={
            "role": "frozen_pre_update_reference",
            "train_action_uids": sorted(train_uids),
            "source_posterior_uid": source_uid,
        },
    )
    return state, {
        "old_designs": old_designs,
        "old_truth": old_truth,
        "old_prediction": old_prediction,
        "action_uids": action_uids,
        "designs": designs,
        "checkpoint_metadata": checkpoint_metadata,
        "pool": pool_payload,
        "static": static,
        "static_path": static_path,
        "pool_path": pool_path,
    }


def build(
    *,
    contract_path: Path = DEFAULT_CONTRACT,
    output_path: Path = DEFAULT_OUTPUT,
    posterior_path: Path = DEFAULT_POSTERIOR,
    static_freeze_path: Path = DEFAULT_STATIC_FREEZE,
    mechanism_resolution_path: Path = DEFAULT_MECHANISM_RESOLUTION,
    device: str = "cpu",
) -> dict[str, Any]:
    for path in (output_path, posterior_path, static_freeze_path, mechanism_resolution_path):
        if path.exists():
            raise FileExistsError(f"Round-24B freeze output already exists: {path}")
    contract = load_contract(contract_path)
    state, source = recompute_prior(contract, device=device)
    posterior_record = state.to_record()
    write_json(posterior_path, posterior_record)

    reference = contract["reference"]
    c1_checkpoint = repo_path(reference["c1_checkpoint_path"])
    c2_checkpoint = repo_path(reference["c2s_checkpoint_path"])
    c1_identity = json.loads((c1_checkpoint.parent / "run_result.json").read_text(encoding="utf-8"))
    c2_identity = json.loads((c2_checkpoint.parent / "run_result.json").read_text(encoding="utf-8"))
    c1_hash = sha256_file(c1_checkpoint)
    c2_hash = sha256_file(c2_checkpoint)
    if c1_hash != str(c1_identity.get("checkpoint_sha256") or c1_identity.get("checkpoint_hash")):
        raise ValueError("Round-24B C1 checkpoint hash drift")
    if c2_hash != str(c2_identity.get("checkpoint_sha256") or c2_identity.get("checkpoint_hash")):
        raise ValueError("Round-24B C2-S checkpoint hash drift")
    role_path = repo_path(reference["c1_role_binding_path"])
    policy = {
        "policy_uid": str(contract["prospective"]["common_frozen_agent"]),
        "parameters_updated": False,
        "selection_rule": "decision_risk_utility_without_outcome_visibility",
        "policy_sha256": hashlib.sha256(
            json.dumps(
                {
                    "policy": contract["prospective"]["common_frozen_agent"],
                    "rounds": contract["prospective"]["rounds"],
                    "query_without_replacement": contract["prospective"]["query_without_replacement"],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }
    payload = {
        "schema_version": "neuronsifter-round24b-pre-update-reference-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "frozen",
        "pre_update_reference_frozen": True,
        "posterior_uid_pre": state.posterior_uid,
        "posterior_record": {
            "path": rel(posterior_path),
            "sha256": sha256_file(posterior_path),
            "source_posterior_uid": state.source_posterior_uid,
            "schema_version": state.schema_version,
        },
        "twin_checkpoints": {
            "c1_encoder": {"path": rel(c1_checkpoint), "sha256": c1_hash, "capability_uid": "c1-foundation-state-encoder-" + c1_hash[:16]},
            "c2s_world_action": {"path": rel(c2_checkpoint), "sha256": c2_hash, "capability_uid": "c2s-neuronsifter-world-action-" + c2_hash[:16]},
        },
        "checkpoint_hashes_verified": True,
        "agent_policy": policy,
        "agent_parameters_frozen": True,
        "tri_calibration_identity": source["static"]["tri_reliability_state"],
        "evaluation_manifests": {
            "c1": {
                "split_uid": reference["c1_split_uid"],
                "split_role": reference["c1_split_role"],
                "role_binding_path": rel(role_path),
                "role_binding_sha256": sha256_file(role_path),
            },
            "c2s": {
                "pool_id": source["pool"]["pool_id"],
                "path": rel(source["pool_path"]),
                "sha256": sha256_file(source["pool_path"]),
                "arrays_sha256": source["pool"]["arrays"]["sha256"],
                "split_role": reference["c2s_split_role"],
            },
        },
        "tool_capability_set": source["pool"]["registered_tool_execution"],
        "posterior_condition_probes": {
            "c1": posterior_condition_probe(state, channel="c1_encoder_evaluation", scale=float(contract["posterior"]["c1_condition_scale"])),
            "c2s": posterior_condition_probe(state, channel="c2s_world_action_rollout", scale=float(contract["posterior"]["c2s_condition_scale"])),
        },
        "source_static_reference": {"path": rel(source["static_path"]), "sha256": sha256_file(source["static_path"])},
        "runtime": {
            "host": os.uname().nodename,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "device": device,
        },
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
        "protected_holdout_access": 0,
        "publication_comparison_open": False,
        "c2b": {"status": "blocked_insufficient_real_e2_e4"},
    }
    write_json(output_path, payload)
    static_payload = {
        "schema_version": "neuronsifter-round24b-static-reference-freeze-v1",
        "status": "frozen",
        "pre_update_reference": {
            "path": rel(output_path),
            "sha256": sha256_file(output_path),
            "posterior_uid": state.posterior_uid,
        },
        "posterior_record": {
            "path": rel(posterior_path),
            "sha256": sha256_file(posterior_path),
        },
        "checkpoint_hashes_verified": True,
        "agent_parameters_frozen": True,
        "protected_holdout_access": 0,
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
    }
    write_json(static_freeze_path, static_payload)

    round23_mechanism_path = REPO_ROOT / "Output/audits/round23_coupling_mechanism_gate.json"
    round24_resolution_path = repo_path(reference["static_reference_path"]).parent / "round24_round23_mechanism_resolution.json"
    round24_resolution = json.loads(round24_resolution_path.read_text(encoding="utf-8"))
    mechanism_status = str(round24_resolution.get("status", ""))
    if mechanism_status not in {"supported", "partial", "failed"}:
        raise ValueError("Round-24B mechanism classification is not supported/partial/failed")
    mechanism_payload = {
        "schema_version": "neuronsifter-round24b-round23-mechanism-resolution-v1",
        "status": mechanism_status,
        "classification_reused_not_inferred_from_development_gate": True,
        "round23_mechanism_gate": {
            "path": rel(round23_mechanism_path),
            "sha256": sha256_file(round23_mechanism_path),
        },
        "round24_resolution": {
            "path": rel(round24_resolution_path),
            "sha256": sha256_file(round24_resolution_path),
        },
        "basis": round24_resolution.get("mechanism_resolution_basis", round24_resolution.get("basis", {})),
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
        "protected_holdout_access": 0,
    }
    write_json(mechanism_resolution_path, mechanism_payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--posterior-output", default=str(DEFAULT_POSTERIOR))
    parser.add_argument("--static-freeze-output", default=str(DEFAULT_STATIC_FREEZE))
    parser.add_argument("--mechanism-resolution-output", default=str(DEFAULT_MECHANISM_RESOLUTION))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    payload = build(
        contract_path=Path(args.contract).resolve(),
        output_path=Path(args.output).resolve(),
        posterior_path=Path(args.posterior_output).resolve(),
        static_freeze_path=Path(args.static_freeze_output).resolve(),
        mechanism_resolution_path=Path(args.mechanism_resolution_output).resolve(),
        device=args.device,
    )
    print(json.dumps({"status": payload["status"], "posterior_uid_pre": payload["posterior_uid_pre"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
