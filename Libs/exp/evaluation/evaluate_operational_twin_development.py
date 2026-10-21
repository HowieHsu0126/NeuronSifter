"""Evaluate one frozen Operational Twin assignment on open dev/cal worlds.

This evaluator is post-training and development/calibration-only.  It may use
the controlled-world benchmark harness to score predicted trajectories, but it
never serializes oracle state, parameters, optima, or a sealed/test outcome.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.utils.data import DataLoader

from Libs.exp.training.train_operational_twin_devcal import (
    EXPECTED_ACCESS_POLICY,
    EXPECTED_CLAIM_BOUNDARY,
    _action_encoding_for_contract,
    _normalization,
    _normalized,
    _seed_everything,
    build_method_model,
    require_calibration_execution_authority,
    require_assignment_runtime_resource,
    require_operational_twin_training_authorized,
    resolve_operational_twin_run_contract,
)
from Libs.models.modules.world_action_encoder import WorldActionEncoder
from Libs.utils.data.controlled_world_contracts import (
    bind_world_action,
    classify_world_action,
    select_family_config,
    validate_controlled_world_config,
)
from Libs.utils.data.controlled_world_open_package import _materialized_source
from Libs.utils.data.controlled_world_runtime import (
    _calcium_synaptic_true_utility,
    _electrophysiology_true_utility,
    _molecular_pharmacological_true_utility,
    evaluate_harness_true_utility,
    load_harness_oracle,
    simulate_harness_trajectory,
)
from Libs.utils.data.controlled_world_tensorization import (
    ControlledWorldTensorDataset,
    tensorize_manifest,
)
from Libs.utils.evaluation.operational_twin_development_evidence import (
    DEVELOPMENT_EVIDENCE_SCHEMA_VERSION,
    summarize_counterfactual_pairs,
)
from Libs.utils.evaluation.operational_twin_training_protocol import (
    OperationalTwinProtocolError,
    candidate_execution_artifact_paths,
    load_candidate_protocol_artifacts,
    require_assignment_preflight,
    require_candidate_execution_authority,
    resolve_method_execution_contract,
    validate_assignment_runtime_evidence,
    validate_candidate_artifacts,
)
from Libs.utils.evaluation.operational_twin_candidate_execution import (
    resolve_candidate_execution_protocol,
    require_candidate_attempt_ownership,
)
from Libs.utils.evaluation.operational_twin_training_protocol import candidate_attempt_path
from Libs.utils.exp_utils import resolve_project_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-scope",
        choices=("historical-v1", "candidate-lineage"),
        default="historical-v1",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--training-manifest", required=True)
    parser.add_argument("--development-manifest", required=True)
    parser.add_argument("--calibration-manifest", required=True)
    parser.add_argument("--package-receipt", required=True)
    parser.add_argument("--open-data-qualification", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--assignment-uid", required=True)
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--candidate-lineage-uid")
    parser.add_argument("--protocol-uid")
    parser.add_argument("--attempt-uid")
    parser.add_argument("--candidate-manifest-hash")
    parser.add_argument("--assignment-manifest-hash")
    parser.add_argument("--preflight-uid")
    parser.add_argument("--command-catalog-hash")
    parser.add_argument("--candidate-execution-authority")
    parser.add_argument("--candidate-resource-environment-receipt")
    parser.add_argument(
        "--calibration-execution-authority",
        default=argparse.SUPPRESS,
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.PIPE
    ).strip()


def _candidate_trajectory_utility(
    family: Mapping[str, object],
    oracle: Mapping[str, object],
    action_binding: Mapping[str, object],
    trajectory: Sequence[Mapping[str, float]],
) -> float:
    """Score a predicted trajectory inside the harness without exposing oracle fields."""

    family_uid = str(family["family_uid"])
    if oracle.get("authority_role") != "benchmark_harness" or oracle.get("family_uid") != family_uid:
        raise PermissionError("Predicted-trajectory utility is restricted to the benchmark harness.")
    binding = bind_world_action(
        family,
        dict(action_binding["canonical_world_action"]),
        dict(action_binding["controlled_extension"]),
    ).to_dict()
    if binding != dict(action_binding):
        raise ValueError("Predicted-trajectory utility requires the canonical action binding.")
    if family_uid == "electrophysiology_conductance_world":
        utility = _electrophysiology_true_utility(family, oracle, binding, trajectory)
    elif family_uid == "calcium_synaptic_world":
        utility = _calcium_synaptic_true_utility(family, oracle, binding, trajectory)
    elif family_uid == "molecular_pharmacological_causal_world":
        utility = _molecular_pharmacological_true_utility(family, oracle, binding, trajectory)
    else:
        raise ValueError(f"Unsupported controlled-world family: {family_uid}")
    if not math.isfinite(utility):
        raise ValueError("Predicted-trajectory utility is non-finite.")
    return float(utility)


def _trajectory_rows(
    values: torch.Tensor,
    valid_time_mask: torch.Tensor,
    feature_names: Sequence[str],
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for time_index in torch.nonzero(valid_time_mask, as_tuple=False).flatten().tolist():
        rows.append(
            {
                str(name): float(values[time_index, feature_index])
                for feature_index, name in enumerate(feature_names)
            }
        )
    return rows


def _evaluate_role(
    *,
    role: str,
    model: torch.nn.Module,
    tensorized: Mapping[str, object],
    manifest: Mapping[str, object],
    source: Mapping[str, object],
    method_contract: Mapping[str, object],
    assignment_uid: str,
    method_id: str,
    seed: int,
    norm_mean: torch.Tensor,
    norm_std: torch.Tensor,
    device: torch.device,
    calibration_scale: torch.Tensor | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    loader = DataLoader(
        ControlledWorldTensorDataset(tensorized, WorldActionEncoder()),
        batch_size=32,
        shuffle=False,
    )
    manifest_rows = {str(row["world_uid"]): dict(row) for row in manifest["worlds"]}
    feature_names = [str(value) for value in tensorized["feature_names"]]
    rows: list[dict[str, object]] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            inputs, input_mask, target, target_mask = _normalized(
                batch, norm_mean, norm_std, device
            )
            action = _action_encoding_for_contract(batch, method_contract).to(device)
            family_index = batch["family_index"].to(device)
            output = model(inputs, input_mask, action, family_index)
            predictive_mean = output["predictive_mean"]
            predictive_scale = output.get("predictive_scale")
            if bool(method_contract["calibration_eligible"]):
                if not isinstance(predictive_scale, torch.Tensor):
                    raise ValueError("Uncertainty-aware evaluation requires predictive scale.")
                if role == "calibration" and calibration_scale is not None:
                    predictive_scale = predictive_scale * calibration_scale.to(device).index_select(
                        0, family_index
                    )[:, None, None]
            else:
                predictive_scale = None
            physical_mean = predictive_mean * norm_std + norm_mean
            physical_scale = (
                predictive_scale * norm_std.abs()
                if isinstance(predictive_scale, torch.Tensor)
                else None
            )
            for index, world_uid_value in enumerate(batch["world_uid"]):
                world_uid = str(world_uid_value)
                manifest_row = manifest_rows[world_uid]
                selected = target_mask[index]
                error = (predictive_mean[index] - target[index]).masked_select(selected)
                if error.numel() == 0:
                    raise ValueError(f"{world_uid} has no future target values.")
                family_uid = str(manifest_row["family_uid"])
                materialized = _materialized_source(
                    source,
                    family_uid,
                    world_uid=world_uid,
                    lineage_uid=str(manifest_row["latent_generator_lineage_uid"]),
                    seed=int(manifest_row["seed"]),
                )
                family = select_family_config(materialized, family_uid)
                oracle = load_harness_oracle(
                    materialized,
                    family_uid=family_uid,
                    reader_role="benchmark_harness",
                )
                action_binding = dict(manifest_row["action_binding"])
                reference_trajectory = simulate_harness_trajectory(
                    family, oracle, action_binding
                )
                reference_utility = evaluate_harness_true_utility(
                    family,
                    oracle,
                    action_binding,
                    reference_trajectory,
                    reader_role="benchmark_harness",
                )
                raw_observation = batch["observation"][index].to(device)
                predicted_physical = torch.where(
                    input_mask[index], raw_observation, physical_mean[index]
                )
                valid_time_mask = batch["observation_mask"][index].any(dim=1)
                candidate_trajectory = _trajectory_rows(
                    predicted_physical.cpu(), valid_time_mask, feature_names
                )
                predicted_utility = _candidate_trajectory_utility(
                    family, oracle, action_binding, candidate_trajectory
                )
                if isinstance(predictive_scale, torch.Tensor):
                    selected_scale = predictive_scale[index].masked_select(selected)
                    gaussian_nll = float(
                        (
                            0.5 * (error / selected_scale).pow(2)
                            + torch.log(selected_scale)
                        ).mean()
                    )
                    coverage = float((error.abs() <= 1.96 * selected_scale).float().mean())
                    predictive_scale_mean = float(
                        physical_scale[index].masked_select(selected).mean()
                    )
                else:
                    gaussian_nll = None
                    coverage = None
                    predictive_scale_mean = None
                row = {
                    "schema_version": DEVELOPMENT_EVIDENCE_SCHEMA_VERSION,
                    "role": role,
                    "assignment_uid": assignment_uid,
                    "method_id": method_id,
                    "seed": seed,
                    "world_uid": world_uid,
                    "latent_world_uid": str(manifest_row["latent_world_uid"]),
                    "counterfactual_pair_uid": str(manifest_row["counterfactual_pair_uid"]),
                    "family_uid": family_uid,
                    "action_program_uid": str(manifest_row["action_program_uid"]),
                    "support_status": str(dict(manifest_row["support_status"])["status"]),
                    "trajectory_rmse": float(torch.sqrt((error**2).mean())),
                    "trajectory_mae": float(error.abs().mean()),
                    "gaussian_nll": gaussian_nll,
                    "coverage_95": coverage,
                    "predictive_scale_mean": predictive_scale_mean,
                    "reference_utility": float(reference_utility),
                    "predicted_utility": predicted_utility,
                    "oracle_parameter_fields_serialized": False,
                    "oracle_state_fields_serialized": False,
                    "test_outcome_access": 0,
                }
                if any(
                    not math.isfinite(float(value))
                    for value in row.values()
                    if isinstance(value, float)
                ):
                    raise ValueError(f"{world_uid} produced non-finite evaluation metrics.")
                rows.append(row)
    pairs = summarize_counterfactual_pairs(rows, manifest["counterfactual_pairs"])
    return rows, pairs


def _support_rows(
    manifest: Mapping[str, object], source: Mapping[str, object]
) -> list[dict[str, object]]:
    rows = []
    for case in manifest.get("development_support_cases", []):
        family_uid = str(case["family_uid"])
        family = select_family_config(source, family_uid)
        attempted = dict(case["attempted_action"])
        observed = classify_world_action(
            family,
            dict(attempted["canonical_world_action"]),
            dict(attempted["controlled_extension"]),
        ).to_dict()
        expected = str(dict(case["support_status"])["status"])
        expected_abstain = expected != "supported"
        actual_abstain = str(observed["status"]) != "supported"
        rows.append(
            {
                "support_case_uid": str(case["support_case_uid"]),
                "family_uid": family_uid,
                "support_case_kind": str(case["support_case_kind"]),
                "expected_status": expected,
                "observed_status": str(observed["status"]),
                "status_match": str(observed["status"]) == expected,
                "expected_abstain": expected_abstain,
                "observed_abstain": actual_abstain,
                "abstention_compliant": actual_abstain == expected_abstain,
                "learned_ood_score_available": False,
                "support_policy_source": "deterministic_world_action_support_contract",
            }
        )
    return rows


def _role_summary(
    role: str,
    world_rows: Sequence[Mapping[str, object]],
    pair_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    defined_directions = [
        bool(row["effect_direction_correct"])
        for row in pair_rows
        if row.get("effect_direction_correct") is not None
    ]
    return {
        "role": role,
        "world_count": len(world_rows),
        "latent_world_pair_count": len(pair_rows),
        "trajectory_rmse": math.sqrt(
            sum(float(row["trajectory_rmse"]) ** 2 for row in world_rows)
            / len(world_rows)
        ),
        "mean_action_contrast_error": sum(
            float(row["action_contrast_error"]) for row in pair_rows
        )
        / len(pair_rows),
        "mean_intervention_ranking_regret": sum(
            float(row["intervention_ranking_regret"]) for row in pair_rows
        )
        / len(pair_rows),
        "effect_direction_accuracy": (
            sum(defined_directions) / len(defined_directions)
            if defined_directions
            else None
        ),
        "mean_gaussian_nll": (
            sum(float(row["gaussian_nll"]) for row in world_rows)
            / len(world_rows)
            if all(row.get("gaussian_nll") is not None for row in world_rows)
            else None
        ),
        "coverage_95": (
            sum(float(row["coverage_95"]) for row in world_rows) / len(world_rows)
            if all(row.get("coverage_95") is not None for row in world_rows)
            else None
        ),
    }


def _candidate_evaluation_context(
    args: argparse.Namespace,
    config: Mapping[str, object],
    root: Path,
    output_root: Path,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, object] | None,
]:
    """Validate one completed candidate attempt without reopening selection."""

    resolved = resolve_candidate_execution_protocol(config, root)
    candidate, assignment_manifest, command_catalog, preflight = (
        load_candidate_protocol_artifacts(config, root, resolved_context=resolved)
    )
    blockers = validate_candidate_artifacts(
        config,
        root,
        candidate,
        assignment_manifest,
        command_catalog,
        preflight,
        require_attempt_roots_absent=False,
        resolved_context=resolved,
    )
    assignments = [
        dict(row)
        for row in assignment_manifest["assignments"]
        if row.get("assignment_uid") == args.assignment_uid
    ]
    if len(assignments) != 1:
        blockers.append(
            {
                "code": "candidate_evaluation_assignment_invalid",
                "message": "Candidate assignment is not present exactly once.",
            }
        )
        assignment: dict[str, object] = {}
        attempt: dict[str, object] = {}
    else:
        assignment = assignments[0]
        attempts = [
            dict(row)
            for row in assignment["attempts"]
            if row.get("attempt_uid") == args.attempt_uid
        ]
        if len(attempts) != 1:
            blockers.append(
                {
                    "code": "candidate_evaluation_attempt_invalid",
                    "message": "Candidate attempt is not present exactly once.",
                }
            )
            attempt = {}
        else:
            attempt = attempts[0]
    expected = {
        "candidate_lineage_uid": candidate.get("candidate_lineage_uid"),
        "protocol_uid": candidate.get("protocol_uid"),
        "preflight_uid": preflight.get("preflight_uid"),
        "candidate_manifest_hash": candidate.get("candidate_manifest_hash"),
        "assignment_manifest_hash": assignment_manifest.get(
            "assignment_manifest_hash"
        ),
        "command_catalog_hash": command_catalog.get("command_catalog_hash"),
    }
    for field, value in expected.items():
        if getattr(args, field) != value:
            blockers.append(
                {
                    "code": "candidate_evaluation_identity_mismatch",
                    "message": f"Candidate evaluator {field} drifted.",
                }
            )
    if assignment:
        if (
            assignment.get("method_id") != args.method_id
            or int(assignment.get("seed", -1)) != int(args.seed)
        ):
            blockers.append(
                {
                    "code": "candidate_evaluation_assignment_identity_mismatch",
                    "message": "Candidate evaluator method or seed drifted.",
                }
            )
    if attempt:
        expected_root = candidate_attempt_path(root, str(attempt["output_root"]), require_absent=False)
        candidate_attempt_path(root, output_root.relative_to(root).as_posix(), require_absent=False)
        if output_root.resolve() != expected_root:
            blockers.append(
                {
                    "code": "candidate_evaluation_output_root_mismatch",
                    "message": "Candidate evaluator output root differs from the manifest.",
                }
            )
    if blockers:
        raise OperationalTwinProtocolError(
            "; ".join(f"{row['code']}: {row['message']}" for row in blockers)
        )
    owner = require_candidate_attempt_ownership(config, root, resolved, str(attempt["attempt_uid"]))
    status = json.loads(
        (output_root / "terminal_status.json").read_text(encoding="utf-8")
    )
    provenance = json.loads(
        (output_root / "provenance/runtime_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    if status.get("status") != "COMPLETED":
        raise OperationalTwinProtocolError(
            "candidate_evaluation_attempt_not_completed: evaluator requires a completed attempt."
        )
    runtime_blockers = validate_assignment_runtime_evidence(
        assignment,
        status,
        provenance,
        resolved_context=resolved,
    )
    if runtime_blockers:
        raise OperationalTwinProtocolError(
            "; ".join(
                f"{row['code']}: {row['message']}" for row in runtime_blockers
            )
        )
    paths = candidate_execution_artifact_paths(
        dict(resolved.protocol),
        assignment,
        attempt,
    )
    if (
        args.candidate_execution_authority != paths["execution_authority"]
        or args.candidate_resource_environment_receipt
        != paths["resource_environment_receipt"]
    ):
        raise OperationalTwinProtocolError(
            "candidate_evaluation_execution_binding_path_mismatch: evaluator authority paths drifted."
        )
    started_at = datetime.fromisoformat(
        str(status["started_at"]).replace("Z", "+00:00")
    )
    execution_context = require_candidate_execution_authority(
        config,
        root,
        assignment=assignment,
        attempt=attempt,
        candidate_manifest=candidate,
        assignment_manifest=assignment_manifest,
        command_catalog=command_catalog,
        preflight=preflight,
        execution_authority_path=paths["execution_authority"],
        resource_environment_receipt_path=paths[
            "resource_environment_receipt"
        ],
        cuda_available=False,
        cuda_device_count=0,
        torch_version="",
        current_source_commit=owner["source_commit"],
        now=started_at,
        validate_current_runtime=False,
        resolved_context=resolved,
    )
    if provenance.get("candidate_execution_authority") != execution_context[
        "execution_authority"
    ]:
        raise OperationalTwinProtocolError(
            "candidate_evaluation_authority_binding_mismatch: training provenance authority drifted."
        )
    if provenance.get("candidate_resource_environment_receipt") != execution_context[
        "resource_environment_receipt"
    ]:
        raise OperationalTwinProtocolError(
            "candidate_evaluation_resource_binding_mismatch: training provenance receipt drifted."
        )
    supplied_calibration_path = str(
        getattr(args, "calibration_execution_authority", "") or ""
    ).strip()
    if not supplied_calibration_path:
        raise OperationalTwinProtocolError(
            "candidate_evaluation_calibration_authority_missing: candidate evaluation requires the frozen calibration-authority argument."
        )
    protocol = dict(resolved.protocol)
    configured_calibration_path = str(
        dict(protocol["output_paths"])[
            "candidate_calibration_execution_authority"
        ]
    )
    if supplied_calibration_path != configured_calibration_path:
        raise OperationalTwinProtocolError(
            "candidate_evaluation_calibration_authority_path_mismatch: evaluator authority path is not canonical."
        )
    canonical_calibration_path = (root / configured_calibration_path).resolve()
    calibration_binding = require_calibration_execution_authority(
        config,
        root,
        canonical_calibration_path,
        method_id=str(assignment["method_id"]),
        candidate_lineage_uid=str(assignment["candidate_lineage_uid"]),
        protocol_uid=str(assignment["protocol_uid"]),
        resolved_context=resolved,
        source_commit=owner["source_commit"],
    )
    provenance_binding = provenance.get("calibration_execution_authority")
    calibration_eligible = bool(assignment.get("calibration_eligible"))
    if calibration_eligible and (
        calibration_binding is None or provenance_binding != calibration_binding
    ):
        raise OperationalTwinProtocolError(
            "candidate_evaluation_calibration_authority_binding_mismatch: UQ training provenance differs from the validated authority."
        )
    if not calibration_eligible and (
        calibration_binding is not None or provenance_binding is not None
    ):
        raise OperationalTwinProtocolError(
            "candidate_evaluation_calibration_authority_binding_mismatch: "
            "calibration-ineligible methods must retain null authority bindings."
        )
    return (
        assignment,
        attempt,
        provenance,
        execution_context,
        calibration_binding,
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    root = resolve_project_root()
    contract = resolve_operational_twin_run_contract(args, root, require_output=True)
    config = contract["config"]
    cfg = contract["cfg"]
    paths = contract["paths"]
    require_operational_twin_training_authorized(cfg, paths)
    output_root = Path(paths["output_root"])
    candidate_mode = str(getattr(args, "protocol_scope", "historical-v1")) == "candidate-lineage"
    if candidate_mode:
        (
            assignment,
            attempt,
            training_provenance,
            execution_context,
            calibration_authority_binding,
        ) = (
            _candidate_evaluation_context(
                args,
                config,
                root,
                output_root,
            )
        )
        require_assignment_runtime_resource(
            assignment,
            resource_receipt=execution_context["resource_receipt_payload"],
        )
    else:
        assignment = require_assignment_preflight(
            config,
            root,
            assignment_uid=str(args.assignment_uid),
            method_id=str(args.method_id),
            seed=int(args.seed),
            output_root=paths["output_root"],
        )
        attempt = {}
        training_provenance = json.loads(
            (output_root / "provenance/runtime_provenance.json").read_text(
                encoding="utf-8"
            )
        )
        require_assignment_runtime_resource(assignment)
    status = json.loads(
        (output_root / "terminal_status.json").read_text(encoding="utf-8")
    )
    if status.get("status") != "COMPLETED":
        raise ValueError(f"Assignment {args.assignment_uid} is not terminal COMPLETED.")
    method_contract = resolve_method_execution_contract(
        str(assignment["method_id"]), str(assignment["execution_contract"])
    )
    _seed_everything(int(args.seed), True)
    tensorized = {
        role: tensorize_manifest(paths[role], role, config, root)
        for role in ("training", "development", "calibration")
    }
    manifests = {
        role: json.loads(paths[role].read_text(encoding="utf-8"))
        for role in ("development", "calibration")
    }
    encoder = WorldActionEncoder()
    resource_profile = str(assignment["resource_profile"])
    device = torch.device("cuda" if resource_profile == "gpu_single" else "cpu")
    if bool(method_contract["checkpoint_creation_allowed"]):
        checkpoint_path = output_root / "checkpoints/best_dev.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("assignment_uid") != args.assignment_uid or int(checkpoint.get("seed", -1)) != int(args.seed):
            raise ValueError("Checkpoint identity does not match the frozen assignment.")
        model = build_method_model(
            len(checkpoint["feature_names"]),
            encoder.output_dim,
            checkpoint["model_config"],
            method_contract,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        norm_mean = checkpoint["normalization_mean"]
        norm_std = checkpoint["normalization_std"]
        calibration_scale = checkpoint.get("calibration_scale")
        checkpoint_hash = _sha256(checkpoint_path)
    else:
        checkpoint_paths = list((output_root / "checkpoints").glob("*.pt")) if (output_root / "checkpoints").exists() else []
        if candidate_mode and checkpoint_paths:
            raise OperationalTwinProtocolError(
                "candidate_evaluation_no_fit_checkpoint_present: no-fit assignment must remain checkpoint-free."
            )
        model = build_method_model(
            len(tensorized["training"]["feature_names"]),
            encoder.output_dim,
            cfg["model"],
            method_contract,
        )
        norm_mean, norm_std = _normalization(tensorized["training"]["records"])
        calibration_scale = None
        checkpoint_hash = None
    model = model.to(device)
    norm_mean = norm_mean.to(device)
    norm_std = norm_std.to(device)
    if isinstance(calibration_scale, torch.Tensor):
        calibration_scale = calibration_scale.to(device)
    source = validate_controlled_world_config(config)

    role_outputs: dict[str, dict[str, object]] = {}
    for role in ("development", "calibration"):
        world_rows, pair_rows = _evaluate_role(
            role=role,
            model=model,
            tensorized=tensorized[role],
            manifest=manifests[role],
            source=source,
            method_contract=method_contract,
            assignment_uid=str(args.assignment_uid),
            method_id=str(args.method_id),
            seed=int(args.seed),
            norm_mean=norm_mean,
            norm_std=norm_std,
            device=device,
            calibration_scale=calibration_scale,
        )
        _write_jsonl(output_root / f"evaluation/{role}_world_results.jsonl", world_rows)
        _write_jsonl(output_root / f"evaluation/{role}_pair_results.jsonl", pair_rows)
        role_outputs[role] = _role_summary(role, world_rows, pair_rows)

    support_rows = _support_rows(manifests["development"], source)
    _write_jsonl(output_root / "evaluation/support_abstention_results.jsonl", support_rows)
    support_summary = {
        "case_count": len(support_rows),
        "status_accuracy": sum(bool(row["status_match"]) for row in support_rows) / len(support_rows),
        "abstention_compliance": sum(bool(row["abstention_compliant"]) for row in support_rows) / len(support_rows),
        "learned_ood_score_available": False,
        "support_policy_source": "deterministic_world_action_support_contract",
    }
    summary = {
        "schema_version": DEVELOPMENT_EVIDENCE_SCHEMA_VERSION,
        "status": "PASS",
        "assignment_uid": str(args.assignment_uid),
        "method_id": str(args.method_id),
        "seed": int(args.seed),
        "development_only": True,
        "calibration_world_only": True,
        "formal_qualification": False,
        "sufficient_training": False,
        "twin_qualification_pass": False,
        "roles": role_outputs,
        "support_abstention": support_summary,
        "training_code_commit": training_provenance.get("training_code_commit", training_provenance.get("git_commit")),
        "evaluation_code_commit": _git_commit(root),
        "checkpoint_sha256": checkpoint_hash,
        "device_provenance": {
            "hostname": os.uname().nodename,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
            "slurm_gpus_on_node": os.environ.get("SLURM_GPUS_ON_NODE"),
            "device_type": device.type,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        },
        "harness_scoring": {
            "oracle_access_role": "benchmark_harness",
            "oracle_parameter_fields_serialized": False,
            "oracle_state_fields_serialized": False,
            "reference_utility_serialized": True,
            "reference_utility_use": "post_training_open_development_calibration_scoring_only",
        },
        "access_flags": EXPECTED_ACCESS_POLICY,
        "claim_flags": EXPECTED_CLAIM_BOUNDARY,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if candidate_mode:
        summary.update(
            {
                "candidate_lineage_uid": args.candidate_lineage_uid,
                "protocol_uid": args.protocol_uid,
                "attempt_uid": args.attempt_uid,
                "candidate_manifest_hash": args.candidate_manifest_hash,
                "assignment_manifest_hash": args.assignment_manifest_hash,
                "preflight_uid": args.preflight_uid,
                "command_catalog_hash": args.command_catalog_hash,
                "selected_checkpoint_role": (
                    "best_dev" if checkpoint_hash is not None else "no_fit"
                ),
                "checkpoint_reselection_performed": False,
                "candidate_execution_authority": training_provenance.get(
                    "candidate_execution_authority"
                ),
                "calibration_execution_authority": calibration_authority_binding,
                "candidate_resource_environment_receipt": training_provenance.get(
                    "candidate_resource_environment_receipt"
                ),
                "attempt_output_root": attempt.get("output_root"),
            }
        )
    _write_json(output_root / "evaluation/summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    try:
        summary = run(args)
    except Exception as error:
        output_root = Path(args.output_root)
        if not output_root.is_absolute():
            output_root = resolve_project_root() / output_root
        if output_root.exists():
            failure_path = output_root / "evaluation/failures.jsonl"
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            with failure_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "assignment_uid": args.assignment_uid,
                            "method_id": args.method_id,
                            "seed": args.seed,
                            "error_type": type(error).__name__,
                            "error": str(error),
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        raise
    print(
        json.dumps(
            {
                "status": summary["status"],
                "assignment_uid": summary["assignment_uid"],
                "development_world_count": summary["roles"]["development"]["world_count"],
                "development_pair_count": summary["roles"]["development"]["latent_world_pair_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
