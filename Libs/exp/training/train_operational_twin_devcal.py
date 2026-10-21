"""Train the bounded controlled-world Operational Twin development/calibration slice."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
from typing import Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch.utils.data import DataLoader
import yaml

from Libs.exp.training.pretrain_microenvironment_world_model import save_portable_training_checkpoint
from Libs.models.modules.world_action_encoder import WORLD_ACTION_FIELDS, WorldActionEncoder, world_action_schema_hash
from Libs.models.networks.operational_twin import (
    LinearDeltaOperationalTwin,
    MeanOnlyOperationalTwin,
    OperationalTwin,
    PersistenceNoActionPredictor,
)
from Libs.utils.data.controlled_world_contracts import FAMILY_UIDS
from Libs.utils.data.controlled_world_open_package import reject_forbidden_path_tokens
from Libs.utils.data.controlled_world_tensorization import ControlledWorldTensorDataset, tensorize_manifest
from Libs.utils.evaluation.operational_twin_candidate_execution import (
    claim_candidate_attempt,
    require_candidate_calibration_source_bindings,
    require_future_execution_selection,
    resolve_candidate_execution_protocol,
)
from Libs.utils.evaluation.operational_twin_training_protocol import (
    CANDIDATE_FORBIDDEN_STATE_INPUT_FIELDS,
    CANDIDATE_RESOURCE_PROFILES,
    OperationalTwinProtocolError,
    ResolvedCandidateExecutionProtocol,
    candidate_attempt_path,
    candidate_source_identity_hash,
    candidate_runtime_slurm_identity,
    candidate_identity_payload,
    compute_candidate_calibration_authority_uid,
    repo_file_identity,
    require_assignment_preflight,
    require_candidate_assignment_preflight,
    require_candidate_execution_authority,
    resolve_method_execution_contract,
)
from Libs.utils.exp_utils import load_config_stack, resolve_project_root


CANONICAL_CONFIG_STACK = Path("Libs/configs/config_stack.yaml")
CANONICAL_MODEL_CONFIG = Path("Libs/configs/model/operational_twin.yaml")

DEFAULT_CALIBRATION_EXECUTION_AUTHORITY = Path(
    "Output/audits/operational_twin_calibration_execution_authority.json"
)
EXPECTED_ACCESS_POLICY = {
    "six_context_outcome_access": 0,
    "protected_holdout_access": 0,
    "test_outcome_access": 0,
    "tool_access": 0,
    "mcp_access": 0,
    "agent_trace_access": 0,
    "agent_access": 0,
    "memory_access": 0,
    "coevolution_access": 0,
}
EXPECTED_CLAIM_BOUNDARY = {
    "training_authorized": False,
    "sufficient_training": False,
    "twin_qualification_pass": False,
    "agent_work_open": False,
    "test_manifest_open": False,
    "formal_qualification": False,
    "tool_gate_open": False,
    "mcp_gate_open": False,
    "agent_trace_gate_open": False,
    "memory_gate_open": False,
    "coevolution_gate_open": False,
}
EXPECTED_ROLE_LINEAGE_ISOLATION_FIELDS = [
    "world_uid",
    "latent_world_uid",
    "latent_generator_lineage_uid",
    "seed_identity",
    "replay_identity",
    "observable_output_identity",
    "protocol_state_identity",
]
EXPECTED_ACTION_RESULT_IDENTITY_FIELD = "protocol_state_identity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--calibration-execution-authority",
        default=DEFAULT_CALIBRATION_EXECUTION_AUTHORITY.as_posix(),
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--assignment-uid", required=True)
    parser.add_argument("--candidate-lineage-uid")
    parser.add_argument("--protocol-uid")
    parser.add_argument("--attempt-uid")
    parser.add_argument("--candidate-manifest-hash")
    parser.add_argument("--assignment-manifest-hash")
    parser.add_argument("--preflight-uid")
    parser.add_argument("--command-catalog-hash")
    parser.add_argument("--candidate-execution-authority")
    parser.add_argument("--candidate-resource-environment-receipt")
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-input")
    parser.add_argument("--history-input")
    parser.add_argument("--checkpoint-input")
    parser.add_argument("--optimizer-state-input")
    parser.add_argument("--model-state-input")
    parser.add_argument("--calibration-state-input")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_hash(value: object) -> str:
    return _sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode())


def _seed_everything(seed: int, deterministic: bool) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)


def _resolve_path(root: Path, value: str | Path, *, must_exist: bool) -> Path:
    path = Path(value)
    candidate = path if path.is_absolute() else root / path
    return candidate.resolve(strict=must_exist)


def _require_canonical_path(
    root: Path,
    supplied: str | Path,
    configured: str | Path,
    label: str,
    *,
    must_exist: bool,
) -> Path:
    supplied_path = _resolve_path(root, supplied, must_exist=must_exist)
    configured_path = _resolve_path(root, configured, must_exist=must_exist)
    if supplied_path != configured_path:
        raise ValueError(
            f"{label} must resolve to the canonical config-stack path "
            f"{configured_path}; received {supplied_path}."
        )
    return configured_path


def _require_canonical_declared_path(
    root: Path,
    supplied: str | Path,
    configured: str | Path,
    label: str,
    *,
    must_exist: bool,
) -> Path:
    reject_forbidden_path_tokens(supplied, label=label)
    path = _require_canonical_path(root, supplied, configured, label, must_exist=must_exist)
    if must_exist and not path.is_file():
        raise ValueError(f"{label} does not exist: {path}")
    return path


def _git_output(root: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=root, text=True, stderr=subprocess.PIPE
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(f"Unable to bind canonical receipt identity through Git: {arguments}") from error


def tracked_reference_identity(root: Path, path: Path) -> dict[str, object]:
    """Bind a receipt through the committed Git tree without opening its content."""
    relative_path = path.relative_to(root).as_posix()
    git_commit = _git_output(root, "rev-parse", "--verify", "HEAD^{commit}")
    object_id = _git_output(root, "rev-parse", "--verify", f"{git_commit}:{relative_path}")
    return {
        "path": relative_path,
        "authority": "git_commit_tree_blob",
        "git_object_id": object_id,
        "git_object_type": "blob",
        "receipt_content_opened": False,
    }


class OperationalTwinTrainingClosedError(PermissionError):
    """Raised when the independent open-data qualification has not authorized training."""


def require_operational_twin_training_authorized(cfg: Mapping[str, object], paths: Mapping[str, Path]) -> None:
    claim_boundary = dict(cfg.get("claim_boundary", {}))
    if claim_boundary != EXPECTED_CLAIM_BOUNDARY:
        raise OperationalTwinTrainingClosedError("Operational Twin training is closed: downstream config gates are not closed.")
    qualification_path = paths.get("future_open_data_qualification")
    if qualification_path is None or not qualification_path.is_file():
        raise OperationalTwinTrainingClosedError("Operational Twin training is closed: independent open-data qualification is absent.")
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    required_terminal = dict(cfg.get("references", {})).get("required_open_data_qualification_terminal")
    if qualification.get("terminal_decision") != required_terminal:
        raise OperationalTwinTrainingClosedError("Operational Twin training is closed: independent open-data qualification is not positive.")
    if qualification.get("training_authorized") is not True or qualification.get("status") != "PASS":
        raise OperationalTwinTrainingClosedError("Operational Twin training is closed: qualification authority did not authorize training.")
    role_isolation = qualification.get("role_lineage_isolation")
    if not isinstance(role_isolation, Mapping):
        raise OperationalTwinTrainingClosedError("Operational Twin training is closed: qualification role-lineage isolation proof is missing.")
    if role_isolation.get("checked_fields") != EXPECTED_ROLE_LINEAGE_ISOLATION_FIELDS:
        raise OperationalTwinTrainingClosedError("Operational Twin training is closed: qualification role-lineage isolation fields are incomplete.")
    if role_isolation.get("action_result_identity_field") != EXPECTED_ACTION_RESULT_IDENTITY_FIELD:
        raise OperationalTwinTrainingClosedError("Operational Twin training is closed: qualification action-result identity binding is invalid.")
    if role_isolation.get("status") != "PASS":
        raise OperationalTwinTrainingClosedError("Operational Twin training is closed: qualification role-lineage isolation did not pass.")
    if role_isolation.get("overlaps") != []:
        raise OperationalTwinTrainingClosedError("Operational Twin training is closed: qualification role-lineage isolation overlaps are present.")
    closed = dict(qualification.get("closed_downstream_gates", {}))
    expected_closed = dict(EXPECTED_CLAIM_BOUNDARY)
    expected_closed["training_authorized"] = True
    if closed != expected_closed:
        raise OperationalTwinTrainingClosedError("Operational Twin training is closed: qualification authority changed downstream gates.")
    receipt_path = paths.get("open_action_package_receipt")
    if receipt_path is None or not receipt_path.is_file():
        raise OperationalTwinTrainingClosedError("Operational Twin training is closed: open action package receipt is absent.")
    manifest_identities = dict(qualification.get("input_identities", {}))
    manifest_content = dict(qualification.get("input_content_identities", {}))
    config_stack_path = paths.get("config_stack")
    if config_stack_path is None:
        raise OperationalTwinTrainingClosedError("Operational Twin training is closed: config-stack identity is absent.")
    repository_root = config_stack_path.parents[2]
    for role in ("training", "development", "calibration"):
        path = paths.get(role)
        if path is None or not path.is_file():
            raise OperationalTwinTrainingClosedError(f"Operational Twin training is closed: {role} manifest is absent.")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if manifest_identities.get(role, {}).get("path") != path.relative_to(repository_root).as_posix():
            raise OperationalTwinTrainingClosedError(f"Operational Twin training is closed: {role} manifest identity does not match qualification.")
        if manifest_content.get(role) != payload.get("content_identity"):
            raise OperationalTwinTrainingClosedError(f"Operational Twin training is closed: {role} manifest content identity does not match qualification.")


def resolve_operational_twin_run_contract(
    args: argparse.Namespace,
    root: Path,
    *,
    require_output: bool,
) -> dict[str, object]:
    """Resolve every runtime path against the one canonical config-stack contract."""
    repository_root = root.resolve(strict=True)
    config_stack_path = _resolve_path(repository_root, CANONICAL_CONFIG_STACK, must_exist=True)
    model_config_path = _require_canonical_path(
        repository_root,
        args.config,
        CANONICAL_MODEL_CONFIG,
        "Operational Twin config",
        must_exist=True,
    )
    stack_payload = yaml.safe_load(config_stack_path.read_text(encoding="utf-8")) or {}
    includes = stack_payload.get("config_includes", [])
    expected_include = CANONICAL_MODEL_CONFIG.relative_to(CANONICAL_CONFIG_STACK.parent).as_posix()
    expected_open_include = "data/controlled_world_open_action_identification.yaml"
    if not isinstance(includes, list) or expected_include not in includes or expected_open_include not in includes:
        raise ValueError("Canonical config stack does not include the Operational Twin and open action-identification configs.")

    config = load_config_stack(config_stack_path)
    cfg = config.get("operational_twin")
    if not isinstance(cfg, Mapping):
        raise ValueError("Canonical config stack is missing operational_twin configuration.")
    if cfg.get("access_policy") != EXPECTED_ACCESS_POLICY:
        raise ValueError("Operational Twin access policy must remain exactly zero.")
    if cfg.get("claim_boundary") != EXPECTED_CLAIM_BOUNDARY:
        raise ValueError("Operational Twin qualification and Agent-open flags must remain false.")

    configured_manifests = cfg.get("manifests")
    configured_references = cfg.get("references")
    if not isinstance(configured_manifests, Mapping) or not isinstance(configured_references, Mapping):
        raise ValueError("Operational Twin canonical manifest/reference mappings are missing.")
    assignment_mode = bool(
        str(getattr(args, "assignment_uid", "") or "")
        or str(getattr(args, "method_id", "") or "")
        or getattr(args, "seed", None) is not None
    )
    candidate_mode = (
        str(getattr(args, "protocol_scope", "historical-v1"))
        == "candidate-lineage"
    )
    if assignment_mode:
        reject_forbidden_path_tokens(args.output_root, label="Operational Twin output root")
        if candidate_mode:
            output_root = candidate_attempt_path(
                repository_root,
                args.output_root,
                require_absent=False,
            )
            if require_output and not output_root.is_dir():
                raise OperationalTwinProtocolError(
                    "candidate_attempt_output_missing: the manifest-bound attempt root does not exist."
                )
        else:
            output_root = _resolve_path(
                repository_root,
                args.output_root,
                must_exist=require_output,
            )
    else:
        output_root = _require_canonical_path(
            repository_root,
            args.output_root,
            cfg["output_root"],
            "Operational Twin output root",
            must_exist=require_output,
        )
    paths = {
        "config_stack": config_stack_path,
        "model_config": model_config_path,
        "training": _require_canonical_declared_path(
            repository_root,
            args.training_manifest,
            configured_manifests["training"],
            "Training manifest",
            must_exist=False,
        ),
        "development": _require_canonical_declared_path(
            repository_root,
            args.development_manifest,
            configured_manifests["development"],
            "Development manifest",
            must_exist=False,
        ),
        "calibration": _require_canonical_declared_path(
            repository_root,
            args.calibration_manifest,
            configured_manifests["calibration"],
            "Calibration manifest",
            must_exist=False,
        ),
        "open_action_package_receipt": _require_canonical_declared_path(
            repository_root,
            args.package_receipt,
            configured_references["open_action_package_receipt"],
            "Open action package receipt",
            must_exist=False,
        ),
        "future_open_data_qualification": _require_canonical_path(
            repository_root,
            args.open_data_qualification,
            configured_references["future_open_data_qualification"],
            "Future open-data qualification",
            must_exist=False,
        ),
        "calibration_execution_authority": _resolve_path(
            repository_root,
            getattr(
                args,
                "calibration_execution_authority",
                DEFAULT_CALIBRATION_EXECUTION_AUTHORITY,
            ),
            must_exist=False,
        ),
        "output_root": output_root,
    }
    return {"config": config, "cfg": cfg, "paths": paths}


def _normalization(records: list[dict[str, object]]) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.stack([row["observation"] for row in records])
    masks = torch.stack([row["observation_mask"] for row in records])
    count = masks.sum(dim=(0, 1)).clamp_min(1)
    mean = (values * masks).sum(dim=(0, 1)) / count
    variance = (((values - mean) * masks) ** 2).sum(dim=(0, 1)) / count
    return mean, variance.sqrt().clamp_min(1.0e-4)


def _normalized(batch: Mapping[str, object], mean: torch.Tensor, std: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = batch["input_observation"].to(device)
    input_mask = batch["input_mask"].to(device)
    targets = batch["target_observation"].to(device)
    target_mask = batch["target_mask"].to(device)
    normalized_inputs = torch.where(input_mask, (inputs - mean) / std, torch.zeros_like(inputs))
    normalized_targets = torch.where(target_mask, (targets - mean) / std, torch.zeros_like(targets))
    if torch.any(input_mask & target_mask):
        raise ValueError("Causal prefix inputs overlap future targets.")
    return normalized_inputs, input_mask, normalized_targets, target_mask


def masked_mse(mean: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (mean - target).pow(2).masked_select(mask).mean()


def masked_gaussian_nll(mean: torch.Tensor, scale: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    terms = 0.5 * ((target - mean) / scale).pow(2) + torch.log(scale)
    return terms.masked_select(mask).mean()


def loss_for_method(output: Mapping[str, object], target: torch.Tensor, mask: torch.Tensor, method_contract: Mapping[str, object]) -> torch.Tensor:
    mean = output["predictive_mean"]
    if method_contract.get("calibration_eligible") is True:
        scale = output.get("predictive_scale")
        if not isinstance(scale, torch.Tensor):
            raise ValueError("Uncertainty-aware method requires learned predictive scale.")
        return masked_gaussian_nll(mean, scale, target, mask)
    return masked_mse(mean, target, mask)


def build_method_model(feature_count: int, action_dim: int, model_config: Mapping[str, object], method_contract: Mapping[str, object]) -> torch.nn.Module:
    family = str(method_contract["runtime_family"])
    if family == "nontrainable_baseline":
        return PersistenceNoActionPredictor()
    if family == "linear_delta_baseline":
        return LinearDeltaOperationalTwin(feature_count, action_dim)
    if family == "operational_twin_backbone_no_action_ablation":
        return OperationalTwin(feature_count, action_dim, dict(model_config))
    if family == "operational_twin_deterministic_action_conditioned":
        return MeanOnlyOperationalTwin(feature_count, action_dim, dict(model_config))
    if family == "operational_twin_uncertainty_aware_action_conditioned":
        return OperationalTwin(feature_count, action_dim, dict(model_config))
    raise OperationalTwinProtocolError(f"unsupported_runtime_family: {family}")


def method_protocol_config(
    config: Mapping[str, object],
    method_id: str,
    *,
    candidate_lineage: bool = False,
    resolved_context: ResolvedCandidateExecutionProtocol | None = None,
) -> dict[str, object]:
    if candidate_lineage:
        candidate = resolved_context.protocol if resolved_context is not None else config.get("operational_twin_candidate_lineage_protocol")
        protocol = (
            dict(candidate).get("scientific_contract")
            if isinstance(candidate, Mapping)
            else None
        )
    else:
        protocol = config.get("operational_twin_training_protocol")
    if not isinstance(protocol, Mapping):
        raise OperationalTwinProtocolError("missing_training_protocol: selected Operational Twin protocol is absent.")
    matches = [dict(row) for row in protocol.get("methods", []) if dict(row).get("method_id") == method_id]
    if len(matches) != 1:
        raise OperationalTwinProtocolError(f"method_protocol_missing: {method_id} is not present exactly once.")
    return matches[0]


def model_config_for_method(base_config: Mapping[str, object], method_protocol: Mapping[str, object]) -> dict[str, object]:
    merged = dict(base_config)
    capacity = dict(method_protocol.get("capacity", {}))
    for key in ("hidden_dim", "posterior_dim", "depth", "head_count", "mechanism_token_count"):
        if key in capacity:
            merged[key] = capacity[key]
    return merged


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def enforce_capacity(method_protocol: Mapping[str, object], trainable_parameter_count: int) -> None:
    capacity = dict(method_protocol.get("capacity", {}))
    expected = capacity.get("trainable_parameters_expected")
    expected_min = capacity.get("trainable_parameters_expected_min")
    if expected is not None and trainable_parameter_count != int(expected):
        raise OperationalTwinProtocolError(
            f"capacity_mismatch: expected exactly {expected} trainable parameters, observed {trainable_parameter_count}."
        )
    if expected_min is not None and trainable_parameter_count < int(expected_min):
        raise OperationalTwinProtocolError(
            f"capacity_below_protocol_minimum: expected at least {expected_min} trainable parameters, observed {trainable_parameter_count}."
        )


def _scheduler_identity() -> dict[str, object]:
    return candidate_runtime_slurm_identity(
        hostname=os.uname().nodename.split(".", 1)[0]
    )


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, norm_mean: torch.Tensor, norm_std: torch.Tensor, device: torch.device, calibration: torch.Tensor | None = None, method_contract: Mapping[str, object] | None = None) -> dict[str, object]:
    model.eval()
    calibration_eligible = bool(method_contract and method_contract.get("calibration_eligible"))
    family_rows: dict[str, list[tuple[float, float | None, float | None]]] = {name: [] for name in FAMILY_UIDS}
    action_effects: list[float] = []
    action_intensities: list[float] = []
    all_error: list[torch.Tensor] = []
    all_scale: list[torch.Tensor] = []
    dose_index = WORLD_ACTION_FIELDS.index("dose_or_intensity")
    for batch in loader:
        inputs, input_mask, target, mask = _normalized(batch, norm_mean, norm_std, device)
        action = (
            _action_encoding_for_contract(batch, method_contract).to(device)
            if method_contract is not None
            else batch["action_encoding"].to(device)
        )
        family = batch["family_index"].to(device)
        output = model(inputs, input_mask, action, family)
        mean = output["predictive_mean"]
        scale = output.get("predictive_scale") if calibration_eligible else None
        if calibration_eligible and not isinstance(scale, torch.Tensor):
            raise OperationalTwinProtocolError(
                "predictive_scale_missing: uncertainty-aware evaluation requires learned predictive scale."
            )
        if isinstance(scale, torch.Tensor) and calibration is not None:
            scale = scale * calibration.to(device).index_select(0, family)[:, None, None]
        error = (mean - target).abs()
        all_error.append(error.masked_select(mask).cpu())
        finite_scale = scale if isinstance(scale, torch.Tensor) else torch.zeros_like(mean)
        all_scale.append(finite_scale.masked_select(mask).cpu())
        no_action = batch["no_action_encoding"].to(device)
        no_action_output = model(inputs, input_mask, no_action, family)["predictive_mean"]
        action_effects.extend((mean - no_action_output).abs().mean(dim=(1, 2)).cpu().tolist())
        action_intensities.extend(action[:, dose_index].cpu().tolist())
        for index, family_index in enumerate(family.cpu().tolist()):
            selected = mask[index]
            rmse = torch.sqrt(((mean[index] - target[index]).pow(2)).masked_select(selected).mean()).item()
            if calibration_eligible and isinstance(scale, torch.Tensor):
                nll = masked_gaussian_nll(mean[index], scale[index], target[index], selected).item()
                coverage = (((target[index] >= mean[index] - 1.96 * scale[index]) & (target[index] <= mean[index] + 1.96 * scale[index])).masked_select(selected).float().mean().item())
            else:
                nll = None
                coverage = None
            family_rows[FAMILY_UIDS[family_index]].append((rmse, nll, coverage))
    errors = torch.cat(all_error)
    scales = torch.cat(all_scale)
    intensity = torch.tensor(action_intensities)
    effect = torch.tensor(action_effects)
    correlation = 0.0
    if intensity.numel() > 1 and intensity.std() > 0 and effect.std() > 0:
        correlation = float(torch.corrcoef(torch.stack((intensity, effect)))[0, 1])
    return {
        "world_count": len(loader.dataset), "reconstruction_rmse": float(torch.sqrt((errors**2).mean())),
        "mean_absolute_error": float(errors.mean()), "mean_predictive_scale": float(scales.mean()) if calibration_eligible else None,
        "coverage_95": float((errors <= 1.96 * scales).float().mean()) if calibration_eligible else None,
        "loss_contract": str(method_contract.get("loss_name")) if method_contract else "unspecified",
        "calibration_eligible": calibration_eligible,
        "action_contrast_mean": float(effect.mean()), "action_intensity_effect_correlation": correlation,
        "action_ranking_diagnostic": correlation,
        "support_ood_count": int(sum(int(row["support_ood"]) for row in loader.dataset.records)),
        "family_metrics": {
            name: {
                "world_count": len(rows),
                "reconstruction_rmse": sum(row[0] for row in rows) / len(rows),
                "gaussian_nll": (
                    sum(float(row[1]) for row in rows if row[1] is not None) / len(rows)
                    if rows and all(row[1] is not None for row in rows)
                    else None
                ),
                "coverage_95": (
                    sum(float(row[2]) for row in rows if row[2] is not None) / len(rows)
                    if rows and all(row[2] is not None for row in rows)
                    else None
                ),
            }
            for name, rows in family_rows.items()
        },
    }


@torch.no_grad()
def fit_calibration(model: torch.nn.Module, loader: DataLoader, norm_mean: torch.Tensor, norm_std: torch.Tensor, device: torch.device, method_contract: Mapping[str, object] | None = None) -> torch.Tensor:
    if not bool(method_contract and method_contract.get("calibration_eligible")):
        raise OperationalTwinProtocolError("calibration_not_allowed: method is not eligible for UQ calibration fitting.")
    ratios: dict[int, list[torch.Tensor]] = {0: [], 1: [], 2: []}
    model.eval()
    for batch in loader:
        inputs, input_mask, target, mask = _normalized(batch, norm_mean, norm_std, device)
        family = batch["family_index"].to(device)
        action = (
            _action_encoding_for_contract(batch, method_contract).to(device)
            if method_contract is not None
            else batch["action_encoding"].to(device)
        )
        output = model(inputs, input_mask, action, family)
        scale = output.get("predictive_scale")
        if not isinstance(scale, torch.Tensor):
            raise OperationalTwinProtocolError("calibration_scale_missing: calibration requires learned predictive scale.")
        ratio = ((output["predictive_mean"] - target).abs() / scale).detach()
        for index, family_index in enumerate(family.tolist()):
            ratios[family_index].append(ratio[index].masked_select(mask[index]).cpu())
    return torch.tensor([max(float(torch.quantile(torch.cat(ratios[index]), 0.95) / 1.96), 0.05) for index in range(3)])


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _formal_assignment_args(args: argparse.Namespace) -> tuple[str, str, int]:
    assignment_uid = str(getattr(args, "assignment_uid", "") or "")
    method_id = str(getattr(args, "method_id", "") or "")
    seed_arg = getattr(args, "seed", None)
    if not assignment_uid or not method_id or seed_arg is None:
        raise OperationalTwinProtocolError(
            "formal_assignment_required: formal Operational Twin training requires --assignment-uid, --method-id, and --seed."
        )
    return assignment_uid, method_id, int(seed_arg)


def _candidate_state_inputs(args: argparse.Namespace) -> dict[str, object]:
    values = {
        field: getattr(args, field, None)
        for field in CANDIDATE_FORBIDDEN_STATE_INPUT_FIELDS
    }
    if bool(getattr(args, "resume", False)):
        values["resume_input"] = True
    return values


def _require_candidate_runtime_arguments(args: argparse.Namespace) -> dict[str, str]:
    fields = (
        "candidate_lineage_uid",
        "protocol_uid",
        "attempt_uid",
        "candidate_manifest_hash",
        "assignment_manifest_hash",
        "preflight_uid",
        "command_catalog_hash",
        "candidate_execution_authority",
        "candidate_resource_environment_receipt",
    )
    values = {
        field: str(getattr(args, field, "") or "").strip()
        for field in fields
    }
    missing = [field for field, value in values.items() if not value]
    if missing:
        raise OperationalTwinProtocolError(
            "candidate_runtime_identity_missing: candidate-lineage execution requires "
            + ", ".join(missing)
            + "."
        )
    return values


def _failure_class(error: Exception) -> str:
    if isinstance(error, FloatingPointError):
        return "divergence_non_finite"
    if isinstance(error, (InterruptedError, TimeoutError, ConnectionError)):
        return "infrastructure_interruption"
    return "runtime_failure"


def _bind_candidate_runtime_payload(
    payload: dict[str, object],
    runtime_identity: Mapping[str, object] | None,
    initialization: Mapping[str, object] | None,
    execution_context: Mapping[str, object] | None = None,
    output_root: str | None = None,
) -> dict[str, object]:
    if runtime_identity is not None:
        payload.update(runtime_identity)
        payload["fresh_initialization_receipt"] = dict(initialization or {})
        payload["state_resume_used"] = False
        if output_root is not None:
            payload["output_root"] = output_root
        if execution_context is not None:
            payload["candidate_execution_authority"] = dict(
                execution_context["execution_authority"]
            )
            payload["candidate_resource_environment_receipt"] = dict(
                execution_context["resource_environment_receipt"]
            )
    return payload


def _action_encoding_for_contract(batch: Mapping[str, object], method_contract: Mapping[str, object]) -> torch.Tensor:
    action_mode = str(method_contract["action_conditioning"])
    if action_mode in {"no_action_only", "no_action_ablation"}:
        return batch["no_action_encoding"]
    return batch["action_encoding"]


def _write_nontrainable_contract_result(
    output_root: Path,
    *,
    assignment: Mapping[str, object],
    method_contract: Mapping[str, object],
    seed: int,
    runtime_identity: Mapping[str, object] | None = None,
    initialization: Mapping[str, object] | None = None,
    execution_context: Mapping[str, object] | None = None,
    attempt_output_root: str | None = None,
) -> dict[str, object]:
    payload = {
        "schema_version": "neuronsifter-operational-twin-nontrainable-contract-result-v1",
        "assignment_uid": assignment["assignment_uid"],
        "method_id": assignment["method_id"],
        "seed": seed,
        "execution_contract": method_contract["execution_contract"],
        "runtime_family": method_contract["runtime_family"],
        "status": "registered_no_fit_contract",
        "training_executed": False,
        "calibration_fitting_executed": False,
        "checkpoint_created": False,
        "metrics_contract": assignment["metrics_contract"],
        "failure_retention_path": method_contract["failure_retention_path"],
    }
    _bind_candidate_runtime_payload(
        payload,
        runtime_identity,
        initialization,
        execution_context,
        attempt_output_root,
    )
    _append_jsonl(output_root / str(method_contract["metrics_path"]), payload)
    failure_row = {"assignment_uid": assignment["assignment_uid"], "retained_failures": []}
    _bind_candidate_runtime_payload(
        failure_row,
        runtime_identity,
        initialization,
        execution_context,
        attempt_output_root,
    )
    _append_jsonl(output_root / str(method_contract["failure_retention_path"]), failure_row)
    _write_json(output_root / "provenance/runtime_provenance.json", payload)
    return payload


def _write_no_fit_result(
    output_root: Path,
    *,
    assignment: Mapping[str, object],
    method_contract: Mapping[str, object],
    seed: int,
    development_metrics: Mapping[str, object],
    calibration_metrics: Mapping[str, object],
    device_provenance: Mapping[str, object],
    scheduler_identity: Mapping[str, object],
    runtime_identity: Mapping[str, object] | None = None,
    initialization: Mapping[str, object] | None = None,
    execution_context: Mapping[str, object] | None = None,
    attempt_output_root: str | None = None,
) -> dict[str, object]:
    payload = {
        "schema_version": "neuronsifter-operational-twin-no-fit-result-v1",
        "assignment_uid": assignment["assignment_uid"],
        "method_id": assignment["method_id"],
        "seed": seed,
        "execution_contract": method_contract["execution_contract"],
        "runtime_family": method_contract["runtime_family"],
        "loss_contract": method_contract["loss_name"],
        "status": "completed_no_fit_metrics",
        "training_executed": False,
        "calibration_fitting_executed": False,
        "checkpoint_created": False,
        "metrics_contract": assignment["metrics_contract"],
        "failure_retention_path": method_contract["failure_retention_path"],
        "development_metrics": dict(development_metrics),
        "calibration_metrics": dict(calibration_metrics),
        "device_provenance": dict(device_provenance),
        "scheduler_identity": dict(scheduler_identity),
    }
    _bind_candidate_runtime_payload(
        payload,
        runtime_identity,
        initialization,
        execution_context,
        attempt_output_root,
    )
    _append_jsonl(output_root / str(method_contract["metrics_path"]), payload)
    failure_row = {"assignment_uid": assignment["assignment_uid"], "retained_failures": []}
    _bind_candidate_runtime_payload(
        failure_row,
        runtime_identity,
        initialization,
        execution_context,
        attempt_output_root,
    )
    _append_jsonl(output_root / str(method_contract["failure_retention_path"]), failure_row)
    _write_json(output_root / "provenance/runtime_provenance.json", payload)
    return payload


def _write_attempt_status(
    output_root: Path,
    payload: Mapping[str, object],
    *,
    candidate_lineage: bool = False,
) -> None:
    if candidate_lineage:
        _write_json(output_root / "attempt_receipt.json", dict(payload))
    else:
        _write_json(output_root / "attempts" / f"{payload['attempt_uid']}.json", dict(payload))
    _write_json(output_root / "terminal_status.json", dict(payload))


def _next_attempt_uid(output_root: Path, assignment_uid: str) -> str:
    attempts_dir = output_root / "attempts"
    existing = sorted(attempts_dir.glob(f"{assignment_uid}__attempt_*.json")) if attempts_dir.exists() else []
    return f"{assignment_uid}__attempt_{len(existing) + 1:03d}"


def _memory_mebibytes(value: object) -> int | None:
    text = str(value or "").strip().upper()
    if not text:
        return None
    multipliers = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}
    suffix = text[-1]
    if suffix in multipliers:
        number = text[:-1]
        multiplier = multipliers[suffix]
    else:
        number = text
        multiplier = 1
    try:
        parsed = float(number) * multiplier
    except ValueError:
        return None
    if parsed <= 0 or not parsed.is_integer():
        return None
    return int(parsed)


def require_assignment_runtime_resource(
    assignment: Mapping[str, object],
    *,
    resource_receipt: Mapping[str, object] | None = None,
) -> None:
    """Fail closed when a formal assignment is placed on the wrong resource class."""
    resource_profile = str(assignment.get("resource_profile", ""))
    if resource_profile not in {"cpu_light", "gpu_single"}:
        raise OperationalTwinProtocolError(
            f"unsupported_resource_profile: {resource_profile or 'missing'}"
        )
    if not os.environ.get("SLURM_JOB_ID"):
        raise OperationalTwinProtocolError(
            "formal_assignment_requires_slurm: formal assignments must execute on an authorized compute node."
        )
    requirements = assignment.get("resource_requirements")
    if requirements is not None:
        if requirements != CANDIDATE_RESOURCE_PROFILES[resource_profile]:
            raise OperationalTwinProtocolError(
                "candidate_resource_requirements_mismatch: manifest resource requirements drifted."
            )
        runtime_partition = os.environ.get("SLURM_JOB_PARTITION")
        runtime_nodelist = os.environ.get("SLURM_JOB_NODELIST")
        runtime_hostname = os.uname().nodename.split(".", 1)[0]
        if runtime_partition != requirements["partition"]:
            raise OperationalTwinProtocolError(
                "candidate_runtime_partition_mismatch: assignment is not on its exact frozen partition."
            )
        if runtime_nodelist != requirements["nodelist"]:
            raise OperationalTwinProtocolError(
                "candidate_runtime_nodelist_mismatch: assignment is not on its exact frozen nodelist."
            )
        if runtime_hostname != requirements["nodelist"]:
            raise OperationalTwinProtocolError(
                "candidate_runtime_node_mismatch: assignment is not running on its exact frozen node."
            )
        if resource_receipt is None:
            raise OperationalTwinProtocolError(
                "candidate_resource_receipt_missing: exact candidate placement requires a bound receipt."
            )
        if resource_receipt.get("resource_requirements") != requirements:
            raise OperationalTwinProtocolError(
                "candidate_resource_receipt_placement_mismatch: bound receipt resource requirements drifted."
            )
        if resource_receipt.get("resource_allocation") != requirements:
            raise OperationalTwinProtocolError(
                "candidate_resource_receipt_allocation_mismatch: bound receipt allocation drifted."
            )
        if resource_receipt.get("slurm") != _scheduler_identity():
            raise OperationalTwinProtocolError(
                "candidate_resource_receipt_runtime_mismatch: live Slurm identity differs from the bound receipt."
            )
        receipt_environment = resource_receipt.get("environment")
        expected_environment = {
            "hostname": runtime_hostname,
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        if receipt_environment != expected_environment:
            raise OperationalTwinProtocolError(
                "candidate_resource_receipt_environment_mismatch: live environment differs from the bound receipt."
            )
        try:
            runtime_cpus = int(str(os.environ.get("SLURM_CPUS_PER_TASK")))
        except ValueError as error:
            raise OperationalTwinProtocolError(
                "candidate_runtime_cpu_mismatch: SLURM_CPUS_PER_TASK is missing or invalid."
            ) from error
        if runtime_cpus != int(requirements["cpus_per_task"]):
            raise OperationalTwinProtocolError(
                "candidate_runtime_cpu_mismatch: live CPUs per task differ from the frozen profile."
            )
        expected_memory = _memory_mebibytes(requirements["memory"])
        memory_per_node = _memory_mebibytes(os.environ.get("SLURM_MEM_PER_NODE"))
        memory_per_cpu = _memory_mebibytes(os.environ.get("SLURM_MEM_PER_CPU"))
        live_memory = memory_per_node
        if live_memory is None and memory_per_cpu is not None:
            live_memory = memory_per_cpu * runtime_cpus
        if live_memory != expected_memory:
            raise OperationalTwinProtocolError(
                "candidate_runtime_memory_mismatch: live memory differs from the frozen profile."
            )
    if resource_profile == "gpu_single":
        gpu_allocation = next(
            (
                str(os.environ[name])
                for name in ("SLURM_JOB_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_STEP_GPUS")
                if str(os.environ.get(name, "")).strip() not in {"", "0", "(null)", "N/A"}
            ),
            "",
        )
        if not gpu_allocation:
            raise OperationalTwinProtocolError(
                "gpu_single_assignment_requires_slurm_gpu_allocation: the frozen assignment resource profile forbids an unallocated visible GPU or CPU fallback."
            )
        if not torch.cuda.is_available():
            raise OperationalTwinProtocolError(
                "gpu_single_assignment_requires_cuda: the frozen assignment resource profile forbids CPU fallback."
            )
        if torch.cuda.device_count() < 1 or not str(
            os.environ.get("CUDA_VISIBLE_DEVICES", "")
        ).strip():
            raise OperationalTwinProtocolError(
                "gpu_single_assignment_requires_visible_cuda_device: no allocated CUDA device is visible."
            )
        job_gres = str(os.environ.get("SLURM_JOB_GRES", "") or "").strip()
        if job_gres and job_gres not in {"gpu:1", "gres/gpu:1"}:
            raise OperationalTwinProtocolError(
                "candidate_runtime_gres_mismatch: live Slurm GRES differs from gpu:1."
            )
    elif requirements is not None:
        gpu_allocation = any(
            str(os.environ.get(name, "")).strip()
            not in {"", "0", "(null)", "N/A", "None"}
            for name in (
                "SLURM_JOB_GPUS",
                "SLURM_GPUS_ON_NODE",
                "SLURM_STEP_GPUS",
            )
        )
        job_gres = str(os.environ.get("SLURM_JOB_GRES", "") or "").strip()
        cuda_visible = str(os.environ.get("CUDA_VISIBLE_DEVICES", "") or "").strip()
        if (
            job_gres not in {"", "(null)", "N/A", "None"}
            or gpu_allocation
            or torch.cuda.is_available()
            or torch.cuda.device_count() != 0
            or cuda_visible
        ):
            raise OperationalTwinProtocolError(
                "cpu_light_assignment_forbids_gpu: the frozen CPU profile permits neither GPU allocation nor CUDA execution."
            )


def require_calibration_execution_authority(
    config: Mapping[str, object],
    root: Path,
    authority_path: Path,
    *,
    method_id: str,
    candidate_lineage_uid: str | None = None,
    protocol_uid: str | None = None,
    resolved_context: ResolvedCandidateExecutionProtocol | None = None,
    source_commit: str | None = None,
    executing_source_commit: str | None = None,
) -> dict[str, object] | None:
    """Validate the later-round authority required for calibration fitting."""
    method_contract = resolve_method_execution_contract(method_id)
    if not bool(method_contract["calibration_eligible"]):
        return None
    if not authority_path.is_file():
        raise OperationalTwinProtocolError(
            "calibration_execution_authority_missing: calibration fitting is not authorized."
        )
    try:
        authority = json.loads(authority_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise OperationalTwinProtocolError(
            "calibration_execution_authority_invalid: authority is not valid JSON."
        ) from error
    if not isinstance(authority, dict):
        raise OperationalTwinProtocolError(
            "calibration_execution_authority_invalid: authority must be an object."
        )
    candidate_mode = candidate_lineage_uid is not None or protocol_uid is not None
    if candidate_mode and resolved_context is None:
        resolved_context = resolve_candidate_execution_protocol(config, root)
    protocol = (
        dict(resolved_context.protocol)
        if candidate_mode
        else dict(config["operational_twin_training_protocol"])
    )
    candidate_required: dict[str, object] = {}
    if candidate_mode:
        from Libs.utils.evaluation.operational_twin_successor_execution_handoff import _path
        canonical_authority = _path(root, resolved_context.execution["output_paths"]["calibration_execution_authority"])
        if authority_path.absolute() != canonical_authority:
            raise OperationalTwinProtocolError("candidate_calibration_authority_path_mismatch")
        if not candidate_lineage_uid or not protocol_uid:
            raise OperationalTwinProtocolError(
                "calibration_execution_authority_invalid: candidate lineage and protocol identities are both required."
            )
        if (
            candidate_lineage_uid != protocol["candidate_lineage_uid"]
            or protocol_uid != protocol["protocol_uid"]
        ):
            raise OperationalTwinProtocolError(
                "calibration_execution_authority_invalid: candidate lineage or protocol identity drifted."
            )
        schema = dict(
            config["operational_twin_calibration_execution_authority_schema"]
        )
        if set(authority) != set(schema["candidate_required_fields"]):
            raise OperationalTwinProtocolError(
                "calibration_execution_authority_invalid: candidate authority keyset is not exact."
            )
        execution = dict(config["operational_twin_candidate_execution"])
        review = dict(execution["protocol_review_contract"])
        review_path = _resolve_path(root, str(review["path"]), must_exist=True)
        if authority.get("approved_protocol_review_identity") != repo_file_identity(
            root, review_path
        ):
            raise OperationalTwinProtocolError(
                "calibration_execution_authority_invalid: approved protocol review identity drifted."
            )
        frozen_artifacts = dict(execution["frozen_artifacts"])
        preflight_path = _resolve_path(
            root,
            str(dict(frozen_artifacts["preflight_authority"])["path"]),
            must_exist=True,
        )
        try:
            preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise OperationalTwinProtocolError(
                "calibration_execution_authority_invalid: frozen preflight is not valid JSON."
            ) from error
        if not isinstance(preflight, dict):
            raise OperationalTwinProtocolError(
                "calibration_execution_authority_invalid: frozen preflight must be an object."
            )
        if authority.get("successor_execution_binding") != dict(resolved_context.binding):
            raise OperationalTwinProtocolError("calibration_successor_binding_mismatch")
        supplied_selection = authority.get("execution_selection", {})
        selection_path = supplied_selection.get("identities", {}).get("selection_record", {}).get("path", "")
        actual_selection = require_future_execution_selection(config, root, (root / selection_path).parent)
        if supplied_selection != actual_selection:
            raise OperationalTwinProtocolError("calibration_execution_selection_mismatch")
        expected_source_commit = source_commit
        if executing_source_commit is not None and expected_source_commit is None:
            expected_source_commit = authority["source_commit"]
        require_candidate_calibration_source_bindings(
            config, root, authority, resolved_context, actual_selection,
            source_commit=expected_source_commit,
            executing_source_commit=executing_source_commit,
        )
        candidate_required = {
            **schema["candidate_required_boolean_fields"],
            "source_round_id": actual_selection["selection"]["round_id"],
            "preflight_uid": execution["preflight_uid"],
            "candidate_manifest_hash": dict(
                frozen_artifacts["candidate_lineage_manifest"]
            )["internal_hash"],
            "assignment_manifest_hash": dict(
                frozen_artifacts["assignment_manifest"]
            )["internal_hash"],
            "command_catalog_hash": dict(
                frozen_artifacts["assignment_commands"]
            )["internal_hash"],
            "source_identity_hash": candidate_source_identity_hash(preflight),
        }
        if (root / ".git").exists():
            current_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                text=True,
                stderr=subprocess.PIPE,
            ).strip()
            candidate_required["source_commit"] = (
                current_commit if expected_source_commit is None else expected_source_commit
            )
    required = {
        "schema_version": (
            "neuronsifter-operational-twin-candidate-lineage-calibration-execution-authority-v2"
            if candidate_mode
            else "neuronsifter-operational-twin-calibration-execution-authority-v1"
        ),
        "status": "PASS",
        "authority_role": "development_calibration_execution",
        "protocol_uid": protocol["protocol_uid"],
        "calibration_fitting_authorized": True,
        "calibration_world_only": True,
        "checkpoint_selection_role": "development_only",
        "allowed_method_ids": ["full_uncertainty_aware_action_conditioned"],
        "access_flags": EXPECTED_ACCESS_POLICY,
        "claim_flags": EXPECTED_CLAIM_BOUNDARY,
    }
    if candidate_mode:
        required["candidate_lineage_uid"] = protocol["candidate_lineage_uid"]
        required.update(candidate_required)
    if method_id not in authority.get("allowed_method_ids", []):
        raise OperationalTwinProtocolError(
            "calibration_execution_authority_invalid: method is not authorized."
        )
    for key, expected in required.items():
        if authority.get(key) != expected:
            raise OperationalTwinProtocolError(
                f"calibration_execution_authority_invalid: {key} does not match the approved execution contract."
            )
    if candidate_mode and authority.get(
        "authority_uid"
    ) != compute_candidate_calibration_authority_uid(authority):
        raise OperationalTwinProtocolError(
            "calibration_execution_authority_invalid: authority UID does not recompute."
        )
    source_files = authority.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise OperationalTwinProtocolError(
            "calibration_execution_authority_invalid: source file bindings are missing."
        )
    for source in source_files:
        if not isinstance(source, Mapping):
            raise OperationalTwinProtocolError(
                "calibration_execution_authority_invalid: malformed source binding."
            )
        source_path = _resolve_path(root, str(source.get("path", "")), must_exist=True)
        if _sha256_bytes(source_path.read_bytes()) != source.get("sha256"):
            raise OperationalTwinProtocolError(
                f"calibration_execution_authority_stale: source hash drifted for {source_path.relative_to(root)}."
            )
    return {
        "path": authority_path.relative_to(root).as_posix(),
        "sha256": _sha256_bytes(authority_path.read_bytes()),
        "authority_uid": authority.get("authority_uid"),
        "source_round_id": authority.get("source_round_id"),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    root = resolve_project_root()
    contract = resolve_operational_twin_run_contract(args, root, require_output=False)
    config = contract["config"]
    cfg = contract["cfg"]
    paths = contract["paths"]
    output_root = paths["output_root"]
    protocol_scope = str(getattr(args, "protocol_scope", "historical-v1"))
    if protocol_scope not in {"historical-v1", "candidate-lineage"}:
        raise OperationalTwinProtocolError(
            f"unsupported_protocol_scope: {protocol_scope}"
        )
    candidate_mode = protocol_scope == "candidate-lineage"
    resolved = resolve_candidate_execution_protocol(config, root) if candidate_mode else None
    candidate_arguments: dict[str, str] = {}
    runtime_identity: dict[str, object] | None = None
    initialization_receipt: dict[str, object] | None = None
    execution_context: dict[str, object] | None = None
    if candidate_mode:
        candidate_arguments = _require_candidate_runtime_arguments(args)
        setattr(args, "_candidate_attempt_claimed", False)
    elif any(
        str(getattr(args, field, "") or "").strip()
        for field in (
            "candidate_lineage_uid",
            "protocol_uid",
            "attempt_uid",
            "candidate_manifest_hash",
            "assignment_manifest_hash",
            "preflight_uid",
            "command_catalog_hash",
            "candidate_execution_authority",
            "candidate_resource_environment_receipt",
        )
    ):
        raise OperationalTwinProtocolError(
            "candidate_runtime_scope_required: candidate identities are invalid under historical-v1 scope."
        )
    assignment_uid, method_id, seed = _formal_assignment_args(args)
    if candidate_mode:
        candidate_context = require_candidate_assignment_preflight(
            config,
            root,
            candidate_lineage_uid=candidate_arguments["candidate_lineage_uid"],
            protocol_uid=candidate_arguments["protocol_uid"],
            assignment_uid=assignment_uid,
            attempt_uid=candidate_arguments["attempt_uid"],
            method_id=method_id,
            seed=seed,
            output_root=args.output_root,
            candidate_manifest_hash=candidate_arguments["candidate_manifest_hash"],
            assignment_manifest_hash=candidate_arguments["assignment_manifest_hash"],
            command_catalog_hash_value=candidate_arguments["command_catalog_hash"],
            preflight_uid=candidate_arguments["preflight_uid"],
            state_inputs=_candidate_state_inputs(args),
            require_recovery_predecessor=True,
            resolved_context=resolved,
        )
        assignment = dict(candidate_context["assignment"])
        attempt = dict(candidate_context["attempt"])
        runtime_identity = candidate_identity_payload(assignment, attempt)
        supplied_identity = {
            "candidate_lineage_uid": candidate_arguments["candidate_lineage_uid"],
            "protocol_uid": candidate_arguments["protocol_uid"],
            "assignment_uid": assignment_uid,
            "attempt_uid": candidate_arguments["attempt_uid"],
        }
        if runtime_identity != supplied_identity:
            raise OperationalTwinProtocolError(
                "candidate_runtime_identity_mismatch: manifest identities differ from runtime arguments."
            )
        initialization_receipt = dict(candidate_context["initialization"])
        execution_context = require_candidate_execution_authority(
            config,
            root,
            assignment=assignment,
            attempt=attempt,
            candidate_manifest=candidate_context["candidate_manifest"],
            assignment_manifest=candidate_context["assignment_manifest"],
            command_catalog=candidate_context["command_catalog"],
            preflight=candidate_context["preflight"],
            execution_authority_path=candidate_arguments[
                "candidate_execution_authority"
            ],
            resource_environment_receipt_path=candidate_arguments[
                "candidate_resource_environment_receipt"
            ],
            cuda_available=bool(torch.cuda.is_available()),
            cuda_device_count=int(torch.cuda.device_count()),
            torch_version=str(torch.__version__),
            resolved_context=resolved,
        )
        setattr(args, "_candidate_runtime_identity", runtime_identity)
        setattr(args, "_candidate_initialization_receipt", initialization_receipt)
        setattr(args, "_candidate_execution_context", execution_context)
        setattr(args, "_candidate_attempt_output_root", str(attempt["output_root"]))
    else:
        assignment = require_assignment_preflight(
            config,
            root,
            assignment_uid=assignment_uid,
            method_id=method_id,
            seed=seed,
            output_root=output_root,
        )
    require_operational_twin_training_authorized(cfg, paths)
    method_contract = resolve_method_execution_contract(
        str(assignment["method_id"]), str(assignment["execution_contract"])
    )
    method_protocol = method_protocol_config(
        config,
        method_id,
        candidate_lineage=candidate_mode,
        resolved_context=resolved,
    )
    calibration_execution_authority = require_calibration_execution_authority(
        config,
        root,
        paths["calibration_execution_authority"],
        method_id=method_id,
        candidate_lineage_uid=(
            candidate_arguments["candidate_lineage_uid"] if candidate_mode else None
        ),
        protocol_uid=candidate_arguments["protocol_uid"] if candidate_mode else None,
        resolved_context=resolved,
        executing_source_commit=(
            execution_context["resource_receipt_payload"]["source_commit"]
            if candidate_mode else None
        ),
    )
    if getattr(args, "dry_run", False):
        result = {
            "schema_version": "neuronsifter-operational-twin-formal-dry-run-v1",
            "status": "PASS",
            "assignment_uid": assignment_uid,
            "method_id": method_id,
            "seed": seed,
            "output_root": output_root.relative_to(root).as_posix(),
            "execution_contract": method_contract["execution_contract"],
            "budget": dict(method_protocol.get("budget", {})),
            "output_created": output_root.exists(),
        }
        if runtime_identity is not None:
            result.update(runtime_identity)
            result["fresh_initialization_receipt"] = dict(
                initialization_receipt or {}
            )
            result["state_resume_used"] = False
            result["candidate_execution_authority"] = dict(
                execution_context["execution_authority"]
            )
            result["candidate_resource_environment_receipt"] = dict(
                execution_context["resource_environment_receipt"]
            )
        return result
    require_assignment_runtime_resource(
        assignment,
        resource_receipt=(
            execution_context["resource_receipt_payload"]
            if execution_context is not None
            else None
        ),
    )
    started_at = datetime.now(timezone.utc).isoformat()
    if candidate_mode:
        output_root = claim_candidate_attempt(
            config, root, resolved, str(attempt["attempt_uid"]), execution_context,
            claimed_at=datetime.fromisoformat(started_at),
        )
        setattr(args, "_candidate_attempt_claimed", True)
        attempt_uid = candidate_arguments["attempt_uid"]
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        attempt_uid = _next_attempt_uid(output_root, assignment_uid)
    if candidate_mode:
        setattr(args, "_candidate_started_at", started_at)
    _seed_everything(seed, bool(cfg["training"]["deterministic_kernels"]))
    encoder = WorldActionEncoder()
    tensorized = {
        "training": tensorize_manifest(paths["training"], "training", config, root),
        "development": tensorize_manifest(paths["development"], "development", config, root),
        "calibration": tensorize_manifest(paths["calibration"], "calibration", config, root),
    }
    norm_mean, norm_std = _normalization(tensorized["training"]["records"])
    budget = dict(method_protocol.get("budget", {}))
    batch_size = int(budget.get("batch_size", cfg["training"]["batch_size"]))
    loaders = {name: DataLoader(ControlledWorldTensorDataset(value, encoder), batch_size=batch_size, shuffle=(name == "training"), generator=torch.Generator().manual_seed(seed)) for name, value in tensorized.items()}
    resource_profile = str(assignment["resource_profile"])
    device = torch.device("cuda" if resource_profile == "gpu_single" else "cpu")
    device_provenance = {
        "resource_profile": resource_profile,
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
        "hostname": os.uname().nodename.split(".", 1)[0],
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "slurm_gpus_on_node": os.environ.get("SLURM_GPUS_ON_NODE"),
        "slurm_step_gpus": os.environ.get("SLURM_STEP_GPUS"),
        "device_type": device.type,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    if candidate_mode:
        device_provenance.update(
            {
                "python_version": platform.python_version(),
                "torch_version": str(torch.__version__),
            }
        )
    norm_mean, norm_std = norm_mean.to(device), norm_std.to(device)
    selected_model_config = model_config_for_method(cfg["model"], method_protocol)
    model = build_method_model(len(tensorized["training"]["feature_names"]), encoder.output_dim, selected_model_config, method_contract).to(device)
    trainable_parameter_count = count_trainable_parameters(model)
    enforce_capacity(method_protocol, trainable_parameter_count)
    if not bool(method_contract["checkpoint_creation_allowed"]):
        dev = evaluate(model, loaders["development"], norm_mean, norm_std, device, method_contract=method_contract)
        cal = evaluate(model, loaders["calibration"], norm_mean, norm_std, device, method_contract=method_contract)
        scheduler_identity = _scheduler_identity()
        result = _write_no_fit_result(
            output_root,
            assignment=assignment,
            method_contract=method_contract,
            seed=seed,
            development_metrics=dev,
            calibration_metrics=cal,
            device_provenance=device_provenance,
            scheduler_identity=scheduler_identity,
            runtime_identity=runtime_identity,
            initialization=initialization_receipt,
            execution_context=execution_context,
            attempt_output_root=(str(attempt["output_root"]) if candidate_mode else None),
        )
        status = _bind_candidate_runtime_payload(
            {
                "schema_version": "neuronsifter-operational-twin-assignment-terminal-status-v1",
                "attempt_uid": attempt_uid,
                "assignment_uid": assignment_uid,
                "method_id": method_id,
                "seed": seed,
                "status": "COMPLETED",
                "terminal_reason": "no_fit_metrics_completed",
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "budget": dict(budget),
                "epochs_completed": 0,
                "steps_completed": 0,
                "trainable_parameter_count": trainable_parameter_count,
                "checkpoint_created": False,
                "calibration_fitting_executed": False,
                "resource_profile": resource_profile,
                "device_provenance": device_provenance,
                "scheduler_identity": scheduler_identity,
                "access_flags": cfg["access_policy"],
                "claim_flags": EXPECTED_CLAIM_BOUNDARY,
            },
            runtime_identity,
            initialization_receipt,
            execution_context,
            str(attempt["output_root"]) if candidate_mode else None,
        )
        _write_attempt_status(
            output_root,
            status,
            candidate_lineage=candidate_mode,
        )
        result["terminal_status"] = status
        return result
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise OperationalTwinProtocolError("trainable_method_without_parameters: trainable method has no trainable parameters.")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(budget.get("learning_rate", cfg["training"]["learning_rate"])),
        weight_decay=float(budget.get("weight_decay", cfg["training"]["weight_decay"])),
    )
    best_metric = float("inf")
    best_payload: dict[str, object] | None = None
    metrics_path = output_root / "metrics/dev_metrics.jsonl"
    if metrics_path.exists():
        if candidate_mode:
            raise OperationalTwinProtocolError(
                "candidate_attempt_metric_collision: development metrics already exist in the claimed attempt root."
            )
        metrics_path.unlink()
    max_steps = int(budget.get("max_steps", 0))
    steps_completed = 0
    epochs_completed = 0
    for epoch in range(int(budget["epochs"])):
        model.train()
        losses = []
        for batch in loaders["training"]:
            if max_steps and steps_completed >= max_steps:
                break
            inputs, input_mask, target, mask = _normalized(batch, norm_mean, norm_std, device)
            action_encoding = _action_encoding_for_contract(batch, method_contract).to(device)
            output = model(inputs, input_mask, action_encoding, batch["family_index"].to(device))
            loss = loss_for_method(output, target, mask, method_contract)
            if not torch.isfinite(loss):
                raise FloatingPointError("Operational Twin loss became non-finite.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(budget.get("gradient_clip_norm", cfg["training"]["gradient_clip_norm"])))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps_completed += 1
            losses.append(float(loss.detach().cpu()))
        dev = evaluate(model, loaders["development"], norm_mean, norm_std, device, method_contract=method_contract)
        dev["epoch"] = epoch + 1
        dev["training_nll"] = sum(losses) / len(losses)
        _bind_candidate_runtime_payload(
            dev,
            runtime_identity,
            initialization_receipt,
            execution_context,
            str(attempt["output_root"]) if candidate_mode else None,
        )
        _append_jsonl(metrics_path, dev)
        epochs_completed = epoch + 1
        payload = _bind_candidate_runtime_payload(
            {
                "epoch": epoch + 1,
                "step": steps_completed,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "normalization_mean": norm_mean,
                "normalization_std": norm_std,
                "dev_metrics": dev,
                "model_config": selected_model_config,
                "feature_names": tensorized["training"]["feature_names"],
                "seed": seed,
                "assignment_uid": assignment_uid,
                "method_contract": method_contract,
                "method_protocol": method_protocol,
                "trainable_parameter_count": trainable_parameter_count,
            },
            runtime_identity,
            initialization_receipt,
            execution_context,
            str(attempt["output_root"]) if candidate_mode else None,
        )
        save_portable_training_checkpoint(output_root / "checkpoints/last.pt", payload)
        if float(dev["reconstruction_rmse"]) < best_metric:
            best_metric = float(dev["reconstruction_rmse"])
            best_payload = deepcopy(payload)
            save_portable_training_checkpoint(output_root / "checkpoints/best_dev.pt", best_payload)
        if max_steps and steps_completed >= max_steps:
            break
    if best_payload is None:
        raise RuntimeError("No development checkpoint was selected.")
    model.load_state_dict(best_payload["model_state_dict"])
    calibration_scale = None
    if bool(method_contract["calibration_eligible"]):
        calibration_scale = fit_calibration(model, loaders["calibration"], norm_mean, norm_std, device, method_contract)
        calibration_metrics = evaluate(model, loaders["calibration"], norm_mean, norm_std, device, calibration_scale, method_contract)
        calibration_metrics["calibration_scale_by_family"] = {name: float(calibration_scale[index]) for index, name in enumerate(FAMILY_UIDS)}
    else:
        calibration_metrics = evaluate(model, loaders["calibration"], norm_mean, norm_std, device, method_contract=method_contract)
        calibration_metrics["calibration_scale_by_family"] = None
    _bind_candidate_runtime_payload(
        calibration_metrics,
        runtime_identity,
        initialization_receipt,
        execution_context,
        str(attempt["output_root"]) if candidate_mode else None,
    )
    calibration_path = output_root / "metrics/calibration_metrics.jsonl"
    if candidate_mode and calibration_path.exists():
        raise OperationalTwinProtocolError(
            "candidate_attempt_metric_collision: calibration metrics already exist in the claimed attempt root."
        )
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    calibration_path.write_text(json.dumps(calibration_metrics, sort_keys=True) + "\n", encoding="utf-8")
    best_payload["calibration_scale"] = calibration_scale
    best_payload["calibration_metrics"] = calibration_metrics
    save_portable_training_checkpoint(output_root / "checkpoints/best_dev.pt", best_payload)
    resolved = yaml.safe_dump(config, sort_keys=True)
    (output_root / "config.resolved.yaml").write_text(resolved, encoding="utf-8")
    hashes = output_root / "hashes"
    hashes.mkdir(parents=True, exist_ok=True)
    config_hash = _sha256_bytes(resolved.encode())
    (hashes / "config_hash.txt").write_text(config_hash + "\n")
    manifest_hashes = {name: value["manifest_hash"] for name, value in tensorized.items()}
    _write_json(hashes / "data_manifest_hashes.json", manifest_hashes)
    seed_hash = _canonical_hash({"seed": seed, "deterministic_kernels": cfg["training"]["deterministic_kernels"]})
    (hashes / "seed_hash.txt").write_text(seed_hash + "\n")
    schema_hash = world_action_schema_hash()
    (hashes / "world_action_schema_hash.txt").write_text(schema_hash + "\n")
    checkpoint_hash = hashlib.sha256((output_root / "checkpoints/best_dev.pt").read_bytes()).hexdigest()
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unavailable"
    receipt_identities = {
        "open_action_package_receipt": tracked_reference_identity(root, paths["open_action_package_receipt"]),
        "open_data_qualification": tracked_reference_identity(root, paths["future_open_data_qualification"]),
    }
    canonical_paths = {
        name: path.relative_to(root).as_posix()
        for name, path in paths.items()
        if name != "output_root"
    }
    canonical_paths["output_root"] = paths["output_root"].relative_to(root).as_posix()
    provenance = {
        "schema_version": "neuronsifter-operational-twin-runtime-provenance-v2", "created_at": datetime.now(timezone.utc).isoformat(),
        "lineage": {
            name: {
                "manifest_uid": value["manifest"]["manifest_uid"],
                "world_count": len(value["records"]),
            }
            for name, value in tensorized.items()
        },
        "opened_manifest_hashes": manifest_hashes,
        "canonical_path_bindings": canonical_paths,
        "reference_receipt_identities": receipt_identities,
        "world_action_schema_hash": schema_hash,
        "model_config_hash": config_hash, "checkpoint_hash": checkpoint_hash, "seed_hash": seed_hash, "git_commit": git_commit,
        "access_flags": cfg["access_policy"], "development_metrics": best_payload["dev_metrics"], "calibration_metrics": calibration_metrics,
        "calibration_execution_authority": calibration_execution_authority,
        "device_provenance": device_provenance,
        "assignment_uid": assignment_uid,
        "method_id": method_id,
        "seed": seed,
        "method_contract": method_contract,
        "method_protocol": method_protocol,
        "budget": budget,
        "epochs_completed": epochs_completed,
        "selected_checkpoint_epoch": int(best_payload["epoch"]),
        "steps_completed": steps_completed,
        "trainable_parameter_count": trainable_parameter_count,
        "scheduler_identity": _scheduler_identity(),
        "claim_flags": EXPECTED_CLAIM_BOUNDARY,
        "graph_scope": "observable-variable adjacency compatibility path; no anatomical or region-graph behavior is claimed",
        "known_limitations": ["NOT sufficient training", "NOT Twin PASS", "NOT formal qualification", "Agent work remains closed", "test manifest remains unopened", "single seed and bounded capacity/budget", "observable-variable graph only; no anatomical graph claim"],
        "later_qualification_requirements": ["adequate independent-world and family scale", "justified capacity and optimization budget", "convergence and stability diagnostics", "multiple seeds", "five baselines", "development-only selection", "calibration-only uncertainty calibration", "OOD and support checks", "checkpoint/config/data/seed freeze and deterministic replay", "frozen first-use test plan"],
    }
    _bind_candidate_runtime_payload(
        provenance,
        runtime_identity,
        initialization_receipt,
        execution_context,
        str(attempt["output_root"]) if candidate_mode else None,
    )
    _write_json(output_root / "provenance/runtime_provenance.json", provenance)
    terminal_status = _bind_candidate_runtime_payload(
        {
            "schema_version": "neuronsifter-operational-twin-assignment-terminal-status-v1",
            "attempt_uid": attempt_uid,
            "assignment_uid": assignment_uid,
            "method_id": method_id,
            "seed": seed,
            "status": "COMPLETED",
            "terminal_reason": "full_budget_completed",
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "budget": budget,
            "epochs_completed": epochs_completed,
            "selected_checkpoint_epoch": int(best_payload["epoch"]),
            "steps_completed": steps_completed,
            "trainable_parameter_count": trainable_parameter_count,
            "checkpoint_created": True,
            "calibration_fitting_executed": bool(method_contract["calibration_eligible"]),
            "resource_profile": resource_profile,
            "device_provenance": device_provenance,
            "scheduler_identity": _scheduler_identity(),
            "access_flags": cfg["access_policy"],
            "claim_flags": EXPECTED_CLAIM_BOUNDARY,
        },
        runtime_identity,
        initialization_receipt,
        execution_context,
        str(attempt["output_root"]) if candidate_mode else None,
    )
    _write_attempt_status(
        output_root,
        terminal_status,
        candidate_lineage=candidate_mode,
    )
    return provenance


def main() -> None:
    args = parse_args()
    try:
        result = run(args)
        if getattr(args, "dry_run", False):
            print(json.dumps(result, sort_keys=True))
    except Exception as error:
        output_value = getattr(args, "output_root", "")
        try:
            root = resolve_project_root()
            candidate_mode = (
                str(getattr(args, "protocol_scope", "historical-v1"))
                == "candidate-lineage"
            )
            output_root: Path | None
            if candidate_mode:
                output_root = None
                if bool(getattr(args, "_candidate_attempt_claimed", False)):
                    try:
                        output_root = candidate_attempt_path(
                            root,
                            str(
                                getattr(
                                    args,
                                    "_candidate_attempt_output_root",
                                    "",
                                )
                            ),
                            require_absent=False,
                        )
                    except OperationalTwinProtocolError:
                        output_root = None
            else:
                output_root = _resolve_path(root, output_value, must_exist=False)
            if getattr(args, "assignment_uid", None) and getattr(args, "method_id", None) and getattr(args, "seed", None) is not None:
                if candidate_mode and output_root is not None:
                    runtime_identity = dict(
                        getattr(args, "_candidate_runtime_identity", {})
                    )
                    initialization_receipt = dict(
                        getattr(args, "_candidate_initialization_receipt", {})
                    )
                    execution_context = dict(
                        getattr(args, "_candidate_execution_context", {})
                    )
                    attempt_output_root = str(
                        getattr(args, "_candidate_attempt_output_root", "")
                    )
                    failure = _bind_candidate_runtime_payload(
                        {
                            "schema_version": "neuronsifter-operational-twin-assignment-terminal-status-v1",
                            "attempt_uid": str(args.attempt_uid),
                            "assignment_uid": str(args.assignment_uid),
                            "method_id": str(args.method_id),
                            "seed": int(args.seed),
                            "status": "FAILED",
                            "failure_class": _failure_class(error),
                            "terminal_reason": type(error).__name__,
                            "error": str(error),
                            "started_at": getattr(
                                args,
                                "_candidate_started_at",
                                None,
                            ),
                            "finished_at": datetime.now(timezone.utc).isoformat(),
                            "scheduler_identity": _scheduler_identity(),
                            "access_flags": EXPECTED_ACCESS_POLICY,
                            "claim_flags": EXPECTED_CLAIM_BOUNDARY,
                        },
                        runtime_identity,
                        initialization_receipt,
                        execution_context,
                        attempt_output_root,
                    )
                    _write_attempt_status(
                        output_root,
                        failure,
                        candidate_lineage=True,
                    )
                    _append_jsonl(
                        output_root
                        / "failures"
                        / f"{args.method_id}_failures.jsonl",
                        failure,
                    )
                # Historical v1 retains its directory-derived attempt receipt behavior.
                elif (
                    not candidate_mode
                    and output_root is not None
                    and output_root.exists()
                ):
                    attempt_uid = _next_attempt_uid(output_root, str(args.assignment_uid))
                    failure = {
                        "schema_version": "neuronsifter-operational-twin-assignment-terminal-status-v1",
                        "attempt_uid": attempt_uid,
                        "assignment_uid": str(args.assignment_uid),
                        "method_id": str(args.method_id),
                        "seed": int(args.seed),
                        "status": "FAILED",
                        "terminal_reason": type(error).__name__,
                        "error": str(error),
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                        "scheduler_identity": _scheduler_identity(),
                        "access_flags": EXPECTED_ACCESS_POLICY,
                        "claim_flags": EXPECTED_CLAIM_BOUNDARY,
                    }
                    _write_attempt_status(output_root, failure)
                    _append_jsonl(output_root / "failures" / f"{args.method_id}_failures.jsonl", failure)
        finally:
            raise


if __name__ == "__main__":
    main()
