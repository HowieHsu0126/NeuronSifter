"""Fail-closed tensorization for controlled-world development/calibration training."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import Dataset

from Libs.utils.data.controlled_world_contracts import FAMILY_UIDS, ControlledWorldContractError, action_sequence_identity
from Libs.utils.data.controlled_world_open_package import (
    CLOSED_DOWNSTREAM_FLAGS,
    OPEN_MANIFEST_SCHEMA_VERSION,
    ZERO_ACCESS_COUNTERS,
    reject_forbidden_path_tokens,
)
from Libs.utils.data.controlled_world_runtime import generate_runtime_instance
from Libs.utils.data.trajectory_contracts import canonical_sha256


ALLOWED_ROLES = ("training", "development", "calibration")
CANONICAL_OPEN_MANIFESTS = {
    role: Path(f"Input/processed/controlled_neural_world/open_action_identification/{role}_manifest.json")
    for role in ALLOWED_ROLES
}
ALLOWED_ZERO_ACCESS_COUNTERS = {
    "six_context_outcome_access",
    "protected_holdout_access",
    "test_outcome_access",
    "tool_access",
    "mcp_access",
    "agent_trace_access",
    "agent_access",
    "memory_access",
    "coevolution_access",
}
ALLOWED_CLAIM_BOUNDARY_KEYS = {
    "implementation_level_contract_only",
    "benchmark_qualification_claim",
    "empirical_evidence",
    "therapeutic_efficacy_claim",
    "clinical_validation_claim",
}
ALLOWED_CLOSED_DOWNSTREAM_KEYS = set(CLOSED_DOWNSTREAM_FLAGS)
FORBIDDEN_EXACT_KEYS = {
    "twin_qualification", "test_outcome", "protected_holdout", "six_context_outcome",
    "oracle_outcome", "oracle_state", "hidden_state", "hidden_utility", "true_action_utility",
    "future_observation", "therapeutic_label", "supervised_label", "mechanism_source",
    "agent_outcome", "agent_label",
}
FORBIDDEN_KEY_FRAGMENTS = (
    "test_outcome", "protected_holdout", "six_context_outcome", "oracle_", "hidden_utility",
    "future_observation", "therapeutic_", "supervised_", "mechanism_source",
    "agent_result", "agent_outcome", "agent_decision", "agent_trace", "agent_control",
)
FEATURE_NAMES = (
    "membrane_voltage", "calcium_concentration", "spike_rate",
    "intracellular_calcium", "release_probability", "releasable_vesicle_fraction",
    "postsynaptic_current", "free_drug", "target_bound_drug", "target_occupancy",
    "free_agent_exposure", "biologic_response", "downstream_signal",
    "misfolded_protein_load", "aggregate_burden", "neuroinflammation", "clearance_flux",
)


class ControlledWorldAccessError(ValueError):
    """Raised before model execution when a payload crosses the dev/cal firewall."""


def _assert_zero_access_counters(value: object, path: str) -> None:
    if not isinstance(value, Mapping):
        raise ControlledWorldAccessError(f"Access counters must be a mapping at {path}")
    for raw_key, child in value.items():
        key = str(raw_key)
        if key not in ALLOWED_ZERO_ACCESS_COUNTERS:
            raise ControlledWorldAccessError(f"Undeclared access counter at {path}.{raw_key}")
        if isinstance(child, bool) or child != 0:
            raise ControlledWorldAccessError(f"Access counter must remain exactly zero at {path}.{raw_key}")


def _assert_claim_boundary_metadata(value: object, path: str) -> None:
    if not isinstance(value, Mapping):
        raise ControlledWorldAccessError(f"Claim boundary must be a mapping at {path}")
    for raw_key, child in value.items():
        key = str(raw_key)
        if key not in ALLOWED_CLAIM_BOUNDARY_KEYS or not isinstance(child, bool):
            raise ControlledWorldAccessError(f"Invalid claim-boundary metadata at {path}.{raw_key}")
        lowered = key.lower()
        if child and (
            lowered in FORBIDDEN_EXACT_KEYS
            or any(fragment in lowered for fragment in FORBIDDEN_KEY_FRAGMENTS)
            or key != "implementation_level_contract_only"
        ):
            raise ControlledWorldAccessError(f"Forbidden true claim-boundary flag at {path}.{raw_key}")


def _assert_no_forbidden_fields(value: object, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).lower()
            if key == "access_counters":
                _assert_zero_access_counters(child, f"{path}.{raw_key}")
                continue
            if key == "claim_boundary":
                _assert_claim_boundary_metadata(child, f"{path}.{raw_key}")
                continue
            if key == "closed_downstream_flags":
                if not isinstance(child, Mapping) or set(child) != ALLOWED_CLOSED_DOWNSTREAM_KEYS:
                    raise ControlledWorldAccessError(f"Invalid closed downstream flags at {path}.{raw_key}")
                if any(value is not False for value in child.values()):
                    raise ControlledWorldAccessError(f"Downstream flag opened at {path}.{raw_key}")
                continue
            if path.endswith(".access_counters") and str(raw_key) in ALLOWED_ZERO_ACCESS_COUNTERS:
                _assert_zero_access_counters(value, path)
                return
            if key in FORBIDDEN_EXACT_KEYS or any(fragment in key for fragment in FORBIDDEN_KEY_FRAGMENTS):
                raise ControlledWorldAccessError(f"Forbidden controlled-world field at {path}.{raw_key}")
            _assert_no_forbidden_fields(child, f"{path}.{raw_key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_forbidden_fields(child, f"{path}[{index}]")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_bytes(path: str | Path, expected_role: str) -> bytes:
    manifest_path = Path(path)
    try:
        reject_forbidden_path_tokens(manifest_path, label=f"{expected_role}_manifest")
    except ControlledWorldContractError as exc:
        raise ControlledWorldAccessError(exc.message) from exc
    if expected_role not in ALLOWED_ROLES:
        raise ControlledWorldAccessError(f"Unsupported manifest role: {expected_role}")

    repository_root = Path(__file__).resolve().parents[3]
    canonical_path = (repository_root / CANONICAL_OPEN_MANIFESTS[expected_role]).resolve()
    supplied_path = manifest_path if manifest_path.is_absolute() else repository_root / manifest_path
    if supplied_path.resolve() != canonical_path:
        raise ControlledWorldAccessError("Manifest is not the canonical open action-identification path.")
    if not canonical_path.is_file():
        raise ControlledWorldAccessError("Canonical open action-identification manifest is absent.")
    return canonical_path.read_bytes()


def controlled_manifest_sha256(path: str | Path, expected_role: str) -> str:
    """Hash only an explicitly allowed open manifest, including sparse-worktree blobs."""
    return hashlib.sha256(_manifest_bytes(path, expected_role)).hexdigest()


def validate_controlled_manifest_payload(payload: object, expected_role: str) -> dict[str, Any]:
    """Validate an already opened train/development/calibration manifest payload."""
    if expected_role not in ALLOWED_ROLES:
        raise ControlledWorldAccessError(f"Unsupported manifest role: {expected_role}")
    if not isinstance(payload, Mapping):
        raise ControlledWorldAccessError("Controlled-world manifest payload must be a mapping.")
    _assert_no_forbidden_fields(payload)
    if payload.get("schema_version") != OPEN_MANIFEST_SCHEMA_VERSION or payload.get("role") != expected_role:
        raise ControlledWorldAccessError("Controlled-world manifest schema or role mismatch.")
    if payload.get("evidence_class") != "controlled_world_non_empirical":
        raise ControlledWorldAccessError("Only controlled-world open action-identification manifests are allowed.")
    if payload.get("access_counters") != ZERO_ACCESS_COUNTERS:
        raise ControlledWorldAccessError("Manifest access counters are not exactly zero.")
    if payload.get("closed_downstream_flags") != CLOSED_DOWNSTREAM_FLAGS:
        raise ControlledWorldAccessError("Manifest downstream gates are not exactly closed.")
    if tuple(payload.get("family_uids", ())) != tuple(FAMILY_UIDS):
        raise ControlledWorldAccessError("Manifest does not contain the canonical three-family set.")
    if payload.get("independent_unit") != "latent_world":
        raise ControlledWorldAccessError("Open action-identification manifests count independent units by latent world.")
    return dict(payload)


def load_controlled_manifest(path: str | Path, expected_role: str) -> dict[str, Any]:
    payload = json.loads(_manifest_bytes(path, expected_role))
    return validate_controlled_manifest_payload(payload, expected_role)


def _materialize_config(source: Mapping[str, object], row: Mapping[str, object]) -> dict[str, object]:
    value = deepcopy(dict(source))
    family_uid = str(row["family_uid"])
    for family in value["families"]:
        if family["family_uid"] != family_uid:
            continue
        seed = int(row["seed"])
        family["generator"]["seed"] = seed
        family["generator"]["subseeds"] = dict(row["generator_subseeds"])
        family["generator"]["latent_generator_lineage_uid"] = str(row["latent_generator_lineage_uid"])
        family["world"]["world_uid"] = str(row["world_uid"])
        return value
    raise ControlledWorldAccessError(f"Unknown family: {family_uid}")


def tensorize_manifest(path: str | Path, expected_role: str, config: Mapping[str, object], repository_root: Path) -> dict[str, object]:
    manifest = load_controlled_manifest(path, expected_role)
    source = config["controlled_neural_world"]
    samples: list[dict[str, object]] = []
    max_time = 0
    for row in manifest["worlds"]:
        _assert_no_forbidden_fields(row)
        if row.get("role") != expected_role:
            raise ControlledWorldAccessError("Manifest row role violates the open action-identification boundary.")
        if row.get("execution_status") != "executed_supported":
            raise ControlledWorldAccessError("Only supported executed rows can become supervised tensor records.")
        if dict(row.get("support_status", {})).get("status") != "supported":
            raise ControlledWorldAccessError("Only supported rows can become supervised tensor records.")
        materialized = _materialize_config(source, row)
        family_uid = str(row["family_uid"])
        binding = dict(row["action_binding"])
        if row.get("action_binding_identity") != binding.get("binding_uid"):
            raise ControlledWorldAccessError(f"Action binding identity mismatch for {row['world_uid']}")
        if row.get("action_sequence_identity") != action_sequence_identity(binding):
            raise ControlledWorldAccessError(f"Action sequence identity mismatch for {row['world_uid']}")
        instance, _ = generate_runtime_instance(
            materialized,
            family_uid=family_uid,
            repository_root=repository_root,
            config_file_identity=str(row["config_file_identity"]),
            canonical_action=dict(binding["canonical_world_action"]),
            action_extension=dict(binding["controlled_extension"]),
        )
        if canonical_sha256(instance) != row["observable_output_identity"]:
            raise ControlledWorldAccessError(f"Observable replay identity mismatch for {row['world_uid']}")
        if dict(instance["action_binding"]) != binding:
            raise ControlledWorldAccessError(f"Action replay identity mismatch for {row['world_uid']}")
        provenance = dict(instance["generator_provenance"])
        if provenance.get("config_content_identity") != row.get("config_content_identity"):
            raise ControlledWorldAccessError(f"Generator config identity mismatch for {row['world_uid']}")
        _assert_no_forbidden_fields(instance)
        samples.append({"row": dict(row), "instance": instance})
        max_time = max(max_time, len(instance["observations"]))
    feature_index = {name: index for index, name in enumerate(FEATURE_NAMES)}
    records: list[dict[str, object]] = []
    for sample in samples:
        instance = sample["instance"]
        observation = torch.zeros(max_time, len(FEATURE_NAMES), dtype=torch.float32)
        mask = torch.zeros_like(observation, dtype=torch.bool)
        time_values = torch.zeros(max_time, dtype=torch.float32)
        for time_index, time_row in enumerate(instance["observations"]):
            time_values[time_index] = float(time_row["time"]["value"])
            for name, measurement in time_row["measurements"].items():
                if name not in feature_index:
                    raise ControlledWorldAccessError(f"Unregistered observable: {name}")
                observation[time_index, feature_index[name]] = float(measurement["value"])
                mask[time_index, feature_index[name]] = True
        family_index = FAMILY_UIDS.index(instance["world_identity"]["world_family_uid"])
        prefix_length = max(1, max_time // 2)
        prefix_mask = mask.clone()
        prefix_mask[prefix_length:] = False
        future_mask = mask.clone()
        future_mask[:prefix_length] = False
        if not prefix_mask.any() or not future_mask.any():
            raise ControlledWorldAccessError(f"World lacks a causal observed-prefix/future-target split: {sample['row']['world_uid']}")
        records.append({
            "world_uid": sample["row"]["world_uid"], "family_uid": FAMILY_UIDS[family_index],
            "family_index": family_index, "observation": observation, "observation_mask": mask,
            "input_observation": torch.where(prefix_mask, observation, torch.zeros_like(observation)),
            "input_mask": prefix_mask, "target_observation": observation, "target_mask": future_mask,
            "missingness_mask": ~mask, "prefix_length": prefix_length, "time": time_values,
            "support_ood": float(instance["support_status"]["status"] != "supported"),
            "action": instance["action_binding"], "lineage_uid": sample["row"]["latent_generator_lineage_uid"],
            "counterfactual_pair_uid": sample["row"]["counterfactual_pair_uid"],
            "latent_world_uid": sample["row"]["latent_world_uid"],
            "action_program_uid": sample["row"]["action_program_uid"],
        })
    return {
        "manifest": manifest,
        "records": records,
        "development_support_cases": list(manifest.get("development_support_cases", [])),
        "feature_names": list(FEATURE_NAMES),
        "manifest_hash": controlled_manifest_sha256(path, expected_role),
    }


class ControlledWorldTensorDataset(Dataset):
    def __init__(self, tensorized: Mapping[str, object], action_encoder: object) -> None:
        self.records = list(tensorized["records"])
        self.action_encoder = action_encoder

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.records[index]
        return {
            "world_uid": row["world_uid"], "family_index": torch.tensor(row["family_index"], dtype=torch.long),
            "observation": row["observation"], "observation_mask": row["observation_mask"],
            "input_observation": row["input_observation"], "input_mask": row["input_mask"],
            "target_observation": row["target_observation"], "target_mask": row["target_mask"],
            "missingness_mask": row["missingness_mask"], "time": row["time"],
            "support_ood": torch.tensor(row["support_ood"], dtype=torch.float32),
            "action_encoding": self.action_encoder.encode(row["action"], row["family_uid"]),
            "no_action_encoding": self.action_encoder.encode_no_action(row["action"], row["family_uid"]),
        }
