"""Strict contracts for the non-empirical controlled neural world families."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

from Libs.utils.data.action_observation_contracts import (
    InformationAction,
    WorldAction,
    validate_information_action,
    validate_world_action,
)
from Libs.utils.data.trajectory_contracts import canonical_sha256, is_sha256, stable_uid


CONFIG_SCHEMA_VERSION = "neuronsifter-controlled-neural-world-config-v2"
INSTANCE_SCHEMA_VERSION = "neuronsifter-controlled-neural-world-instance-v2"
REPLAY_SCHEMA_VERSION = "neuronsifter-controlled-world-replay-manifest-v1"
FAMILY_MANIFEST_SCHEMA_VERSION = "neuronsifter-controlled-world-family-manifest-v1"
AUDIT_SCHEMA_VERSION = "neuronsifter-controlled-neural-world-validation-v2"

FAMILY_UIDS = (
    "electrophysiology_conductance_world",
    "calcium_synaptic_world",
    "molecular_pharmacological_causal_world",
)
FAMILY_UID_SET = frozenset(FAMILY_UIDS)

WORLD_ACTION_FIELDS = frozenset(WorldAction.__dataclass_fields__)
INFORMATION_ACTION_FIELDS = frozenset(InformationAction.__dataclass_fields__)
ACTION_EXTENSION_FIELDS = frozenset(
    {"modality", "dose_or_intensity", "duration", "combination", "biological_context", "action_semantics"}
)
ACTION_SEMANTICS_FIELDS = frozenset(
    {"target_set", "agent_set", "schedule", "route", "support_domain_ref", "provenance_ref"}
)
RUNTIME_INSTANCE_FIELDS = frozenset(
    {
        "schema_version",
        "authority_role",
        "evidence_class",
        "world_identity",
        "generator_provenance",
        "action_binding",
        "support_status",
        "observations",
        "information_results",
        "information_cost_total",
        "access_counters",
        "claim_boundary",
    }
)
REPLAY_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "replay_uid",
        "world_uid",
        "world_family_uid",
        "latent_generator_lineage_uid",
        "generator_implementation_uid",
        "generator_implementation_version",
        "generator_code_identity",
        "config_content_identity",
        "config_file_identity",
        "seed",
        "seed_identity",
        "action_sequence_identity",
        "information_sequence_identity",
        "observable_output_identity",
        "input_lineage",
        "output_lineage",
        "access_counters",
        "deterministic_reproduction",
    }
)
FAMILY_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "manifest_uid",
        "evidence_class",
        "family_count",
        "family_uids",
        "source_config_ref",
        "source_config_identity",
        "families",
        "access_counters",
        "real_world_formal_agent_evaluation",
        "claim_boundary",
        "benchmark_split_assignment",
        "sealed_test_partitions",
        "information_headroom_qualification",
        "benchmark_qualification",
    }
)
FAMILY_MANIFEST_ROW_FIELDS = (
    "family_uid",
    "world_uid",
    "latent_generator_lineage_uid",
    "seed",
    "seed_identity",
    "runtime_artifact_ref",
    "runtime_content_identity",
    "replay_artifact_ref",
    "replay_content_identity",
    "replay_uid",
)
FORBIDDEN_RUNTIME_KEYS = frozenset(
    {
        "oracle",
        "oracle_state",
        "oracle_parameters",
        "hidden_state",
        "hidden_parameters",
        "hidden_optimum",
        "true_utility",
        "future_observation",
        "future_observations",
        "protected_outcome",
        "protected_outcomes",
    }
)
HIDDEN_REQUESTS = frozenset(
    {
        "oracle_state",
        "oracle_parameters",
        "hidden_state",
        "hidden_parameters",
        "hidden_optimum",
        "true_utility",
        "future_observation",
        "future_observations",
    }
)
PROTECTED_REFERENCE_MARKERS = (
    "six_context",
    "six-context",
    "protected_holdout",
    "protected-holdout",
    "protected_outcome",
)
CLAIM_BOUNDARY = {
    "implementation_level_contract_only": True,
    "benchmark_qualification_claim": False,
    "empirical_evidence": False,
    "therapeutic_efficacy_claim": False,
    "clinical_validation_claim": False,
}

FAMILY_CONTRACTS: dict[str, dict[str, object]] = {
    "electrophysiology_conductance_world": {
        "time_unit": "ms",
        "intervention_type": "electrical_stimulation",
        "modality": "electrical_stimulation",
        "parameter_units": {
            "resting_membrane_voltage": "mV",
            "initial_membrane_voltage": "mV",
            "initial_calcium_concentration": "umol/L",
            "membrane_leak_rate": "per_ms",
            "stimulation_gain": "mV_per_uA_ms",
            "calcium_decay_rate": "per_ms",
            "spike_threshold": "mV",
            "spike_rate_gain": "Hz_per_mV",
            "spike_calcium_gain": "umol_L_per_Hz_ms",
        },
        "observable_units": {
            "membrane_voltage": "mV",
            "calcium_concentration": "umol/L",
            "spike_rate": "Hz",
        },
        "hidden_units": {
            "membrane_leak_rate": "per_ms",
            "stimulation_gain": "mV_per_uA_ms",
            "spike_threshold": "mV",
            "true_action_utility": "utility_unit",
        },
    },
    "calcium_synaptic_world": {
        "time_unit": "ms",
        "intervention_type": "optical_stimulation",
        "modality": "optical_stimulation",
        "parameter_units": {
            "baseline_intracellular_calcium": "umol/L",
            "initial_intracellular_calcium": "umol/L",
            "initial_release_probability": "probability",
            "initial_releasable_vesicle_fraction": "fraction",
            "initial_postsynaptic_current": "pA",
            "calcium_clearance_rate": "per_ms",
            "presynaptic_calcium_gain": "umol_L_per_mW_ms",
            "release_coupling": "per_umol_L",
            "vesicle_release_rate": "per_ms",
            "vesicle_recovery_rate": "per_ms",
            "postsynaptic_current_gain": "pA_per_fraction",
            "postsynaptic_decay_rate": "per_ms",
        },
        "observable_units": {
            "intracellular_calcium": "umol/L",
            "release_probability": "probability",
            "releasable_vesicle_fraction": "fraction",
            "postsynaptic_current": "pA",
        },
        "hidden_units": {
            "calcium_clearance_rate": "per_ms",
            "release_coupling": "per_umol_L",
            "vesicle_recovery_rate": "per_ms",
            "postsynaptic_decay_rate": "per_ms",
            "true_action_utility": "utility_unit",
        },
    },
    "molecular_pharmacological_causal_world": {
        "time_unit": "hour",
        "intervention_type": "pharmacological",
        "modality": "pharmacological",
        "parameter_units": {
            "initial_free_agent_exposure": "nmol/L",
            "initial_target_occupancy": "fraction",
            "initial_downstream_signal": "normalized_signal",
            "initial_biologic_response": "normalized_response",
            "dose_to_exposure_gain": "nmol_L_per_mg_kg_hour",
            "exposure_clearance_rate": "per_hour",
            "binding_affinity": "nmol/L",
            "occupancy_relaxation_rate": "per_hour",
            "signal_activation_rate": "per_hour",
            "signal_decay_rate": "per_hour",
            "biologic_response_rate": "per_hour",
            "biologic_clearance_rate": "per_hour",
            "feedback_rate": "per_hour",
        },
        "observable_units": {
            "free_agent_exposure": "nmol/L",
            "target_occupancy": "fraction",
            "downstream_signal": "normalized_signal",
            "biologic_response": "normalized_response",
        },
        "hidden_units": {
            "binding_affinity": "nmol/L",
            "exposure_clearance_rate": "per_hour",
            "signal_activation_rate": "per_hour",
            "feedback_rate": "per_hour",
            "true_action_utility": "utility_unit",
        },
    },
}


class ControlledWorldContractError(ValueError):
    """A stable, fail-closed controlled-world contract failure."""

    def __init__(self, code: str, message: str, **details: object) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.details = details

    def to_blocker(self) -> dict[str, object]:
        return {"code": self.code, "message": self.message, **self.details}


@dataclass(frozen=True)
class WorldIdentity:
    world_uid: str
    world_family_uid: str
    latent_generator_lineage_uid: str
    biological_context: Mapping[str, str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GeneratorProvenance:
    implementation_uid: str
    implementation_version: str
    implementation_path: str
    code_identity: str
    config_content_identity: str
    config_file_identity: str
    seed: int
    seed_identity: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SupportDecision:
    status: str
    reason_codes: Sequence[str]
    support_domain_ref: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ControlledWorldActionBinding:
    binding_uid: str
    canonical_world_action: Mapping[str, object]
    controlled_extension: Mapping[str, object]
    response_delay_steps: int
    mapping_status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ControlledInformationActionBinding:
    information_action_uid: str
    canonical_information_action: Mapping[str, object]
    outcome_policy: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _mapping(value: object, code: str, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ControlledWorldContractError(code, f"{name} must be a mapping.")
    return dict(value)


def _sequence(value: object, code: str, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ControlledWorldContractError(code, f"{name} must be a list.")
    return list(value)


def _finite(value: object, code: str, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ControlledWorldContractError(code, f"{name} must be finite.", field=name)
    return float(value)


def _exact_keys(value: Mapping[str, object], expected: frozenset[str], code: str, name: str) -> None:
    actual = set(value)
    if actual != set(expected):
        raise ControlledWorldContractError(
            code,
            f"{name} fields must match the contract exactly.",
            missing=sorted(set(expected) - actual),
            unknown=sorted(actual - set(expected)),
        )


def _unit_value(value: object, *, name: str, allow_seed_jitter: bool = False) -> dict[str, object]:
    row = _mapping(value, "invalid_unit_value", name)
    required = {"value", "unit"}
    allowed = required | ({"seed_jitter"} if allow_seed_jitter else set())
    if not required.issubset(row) or set(row) - allowed:
        raise ControlledWorldContractError(
            "unknown_config_field",
            f"{name} fields must match the contract.",
            missing=sorted(required - set(row)),
            unknown=sorted(set(row) - allowed),
        )
    _finite(row["value"], "non_finite_value", f"{name}.value")
    if not isinstance(row["unit"], str) or not row["unit"].strip():
        raise ControlledWorldContractError("invalid_unit", f"{name}.unit must be explicit.")
    if "seed_jitter" in row and _finite(row["seed_jitter"], "non_finite_value", f"{name}.seed_jitter") < 0:
        raise ControlledWorldContractError("invalid_parameter", f"{name}.seed_jitter cannot be negative.")
    return row


def _walk(value: object, prefix: str = "") -> list[tuple[str, str, object]]:
    rows: list[tuple[str, str, object]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            rows.append((path, str(key), item))
            rows.extend(_walk(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            rows.extend(_walk(item, f"{prefix}[{index}]"))
    return rows


def ensure_finite_payload(value: object) -> None:
    for path, _, item in _walk(value):
        if isinstance(item, (int, float)) and not isinstance(item, bool) and not math.isfinite(float(item)):
            raise ControlledWorldContractError("non_finite_value", "NaN and Inf are prohibited.", field=path)


def ensure_no_protected_references(value: object) -> None:
    for path, _, item in _walk(value):
        if isinstance(item, str) and any(marker in item.lower() for marker in PROTECTED_REFERENCE_MARKERS):
            raise ControlledWorldContractError(
                "protected_reference_forbidden",
                "Protected or six-context references are prohibited.",
                field=path,
            )


def ensure_no_runtime_secrets(value: object) -> None:
    for path, key, _ in _walk(value):
        normalized = key.lower()
        if (
            normalized in FORBIDDEN_RUNTIME_KEYS
            or normalized.startswith("oracle_")
            or normalized.startswith("hidden_")
            or normalized.startswith("protected_outcome")
            or normalized.startswith("future_observation")
        ):
            raise ControlledWorldContractError(
                "oracle_access_forbidden",
                "Runtime-visible payload contains a harness-only field.",
                field=path,
            )
    ensure_no_protected_references(value)
    ensure_finite_payload(value)


def validate_zero_access(value: Mapping[str, object]) -> None:
    counters = _mapping(value, "invalid_access_counters", "access counters")
    _exact_keys(
        counters,
        frozenset({"six_context_outcome_access", "protected_holdout_access"}),
        "invalid_access_counters",
        "access counters",
    )
    if counters["six_context_outcome_access"] != 0 or counters["protected_holdout_access"] != 0:
        raise ControlledWorldContractError("protected_access_nonzero", "Controlled-world access counters must remain exactly zero.")


def controlled_world_config(config: Mapping[str, object]) -> dict[str, Any]:
    source = config.get("controlled_neural_world") if "controlled_neural_world" in config else config
    value = _mapping(source, "missing_controlled_world_config", "controlled_neural_world")
    if value.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ControlledWorldContractError("unknown_schema_version", "Controlled-world config schema is not canonical.")
    return value


def select_family_config(config: Mapping[str, object], family_uid: str) -> dict[str, Any]:
    if "family_uid" in config and "generator" in config:
        family = dict(config)
        if family.get("family_uid") != family_uid:
            raise ControlledWorldContractError("family_identity_mismatch", "Family selection does not match the supplied family config.")
        return family
    source = controlled_world_config(config)
    matches = [dict(row) for row in source.get("families", []) if isinstance(row, Mapping) and row.get("family_uid") == family_uid]
    if len(matches) != 1:
        raise ControlledWorldContractError("unsupported_world_family", "Family UID must resolve exactly once.", family_uid=family_uid)
    return matches[0]


def iter_family_configs(config: Mapping[str, object]) -> list[dict[str, Any]]:
    source = controlled_world_config(config)
    return [select_family_config(source, family_uid) for family_uid in FAMILY_UIDS]


def _variable_units(value: object, *, name: str) -> dict[str, str]:
    rows = _sequence(value, "invalid_shape", name)
    result: dict[str, str] = {}
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row, "invalid_shape", f"{name}[{index}]")
        _exact_keys(row, frozenset({"variable", "unit"}), "unknown_config_field", f"{name}[{index}]")
        variable = str(row.get("variable", ""))
        unit = str(row.get("unit", ""))
        if not variable or not unit or variable in result:
            raise ControlledWorldContractError("invalid_shape", f"{name} identities and units must be unique and explicit.")
        result[variable] = unit
    return result


def _validate_family_config(family: Mapping[str, object]) -> dict[str, Any]:
    value = dict(family)
    _exact_keys(
        value,
        frozenset({"family_uid", "generator", "world", "world_action", "information_actions", "support_domain", "output_paths"}),
        "unknown_config_field",
        "controlled-world family",
    )
    family_uid = str(value.get("family_uid", ""))
    if family_uid not in FAMILY_UID_SET:
        raise ControlledWorldContractError("unsupported_world_family", "World family is outside the closed dispatch.", family_uid=family_uid)
    contract = FAMILY_CONTRACTS[family_uid]
    generator = _mapping(value["generator"], "invalid_generator", "generator")
    _exact_keys(
        generator,
        frozenset({"implementation_uid", "implementation_version", "implementation_path", "deterministic", "seed", "seed_policy", "subseeds", "latent_generator_lineage_uid", "parameters"}),
        "unknown_config_field",
        "generator",
    )
    if generator["implementation_path"] != "Libs/utils/data/controlled_world_runtime.py":
        raise ControlledWorldContractError("generator_source_invalid", "Family generator must use the canonical controlled-world runtime.")
    if generator["deterministic"] is not True or generator["seed_policy"] != "fixed_per_world_identity":
        raise ControlledWorldContractError("nondeterministic_generator", "The controlled-world generator must be deterministic.")
    if not isinstance(generator["seed"], int) or isinstance(generator["seed"], bool) or generator["seed"] < 0:
        raise ControlledWorldContractError("invalid_seed", "Generator seed must be a non-negative integer.")
    subseeds = _mapping(generator["subseeds"], "invalid_seed", "generator.subseeds")
    _exact_keys(subseeds, frozenset({"parameter_jitter", "observation_noise"}), "unknown_config_field", "generator.subseeds")
    if any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in subseeds.values()):
        raise ControlledWorldContractError("invalid_seed", "Family sub-seeds must be non-negative integers.")
    if len(set(subseeds.values())) != 2 or generator["seed"] in set(subseeds.values()):
        raise ControlledWorldContractError("duplicate_seed_identity", "Family seed and named sub-seeds must be distinct.")
    parameters = _mapping(generator["parameters"], "invalid_generator", "generator.parameters")
    parameter_units = dict(contract["parameter_units"])
    _exact_keys(parameters, frozenset(parameter_units), "unknown_config_field", "generator.parameters")
    for name, expected_unit in parameter_units.items():
        parameter = _unit_value(parameters[name], name=f"generator.parameters.{name}", allow_seed_jitter=True)
        if parameter["unit"] != expected_unit:
            raise ControlledWorldContractError("invalid_unit", "Generator parameter unit is not canonical.", field=name)

    world = _mapping(value["world"], "invalid_world", "world")
    _exact_keys(
        world,
        frozenset({"world_family_uid", "world_uid", "biological_context", "time_grid", "observable_variables", "hidden_variables", "observation_policy"}),
        "unknown_config_field",
        "world",
    )
    if world["world_family_uid"] != family_uid:
        raise ControlledWorldContractError("family_identity_mismatch", "World and family UIDs must match exactly.")
    context = _mapping(world["biological_context"], "invalid_biological_context", "world.biological_context")
    _exact_keys(context, frozenset({"organism", "region", "compartment", "cell_type"}), "unknown_config_field", "world.biological_context")
    if any(not isinstance(item, str) or not item for item in context.values()):
        raise ControlledWorldContractError("invalid_biological_context", "Biological context must be explicit.")
    time_grid = _mapping(world["time_grid"], "invalid_time_grid", "world.time_grid")
    _exact_keys(time_grid, frozenset({"values", "unit"}), "unknown_config_field", "world.time_grid")
    if time_grid["unit"] != contract["time_unit"]:
        raise ControlledWorldContractError("invalid_unit", "World time unit is not canonical.")
    times = [_finite(item, "invalid_time_grid", "world.time_grid.values") for item in _sequence(time_grid["values"], "invalid_time_grid", "world.time_grid.values")]
    if len(times) < 3 or any(right <= left for left, right in zip(times, times[1:])):
        raise ControlledWorldContractError("invalid_shape", "Time grid must be strictly increasing with at least three points.")
    observable_units = _variable_units(world["observable_variables"], name="world.observable_variables")
    hidden_units = _variable_units(world["hidden_variables"], name="world.hidden_variables")
    if observable_units != contract["observable_units"] or hidden_units != contract["hidden_units"]:
        raise ControlledWorldContractError("invalid_unit", "Family variables and units do not match the closed family contract.")
    observation_policy = _mapping(world["observation_policy"], "invalid_observation_policy", "world.observation_policy")
    _exact_keys(observation_policy, frozenset({"observed_indices", "noise_scale"}), "unknown_config_field", "world.observation_policy")
    observed_indices = _mapping(observation_policy["observed_indices"], "invalid_observation_policy", "observed_indices")
    noise_scale = _mapping(observation_policy["noise_scale"], "invalid_observation_policy", "noise_scale")
    if set(observed_indices) != set(observable_units) or set(noise_scale) != set(observable_units):
        raise ControlledWorldContractError("invalid_shape", "Every observable requires an observation schedule and noise scale.")
    for variable, raw_indices in observed_indices.items():
        indices = _sequence(raw_indices, "invalid_observation_index", f"observed_indices.{variable}")
        if not indices or any(not isinstance(index, int) or isinstance(index, bool) for index in indices):
            raise ControlledWorldContractError("invalid_observation_index", "Observation indices must be non-empty integers.", field=variable)
        if len(indices) != len(set(indices)) or any(index < 0 or index >= len(times) for index in indices):
            raise ControlledWorldContractError("invalid_observation_index", "Observation indices must be unique and in range.", field=variable)
        scale = _unit_value(noise_scale[variable], name=f"noise_scale.{variable}")
        if scale["unit"] != observable_units[variable] or float(scale["value"]) < 0:
            raise ControlledWorldContractError("invalid_unit", "Observation noise must use the observable unit and non-negative scale.", field=variable)
    if all(len(indices) == len(times) for indices in observed_indices.values()):
        raise ControlledWorldContractError("fully_observed_world_forbidden", "The runtime view must remain incomplete.")

    action_spec = _mapping(value["world_action"], "invalid_world_action", "world_action")
    _exact_keys(action_spec, frozenset({"canonical", "extension", "response_delay_steps"}), "unknown_config_field", "world_action")
    action = _mapping(action_spec["canonical"], "invalid_world_action", "world_action.canonical")
    extension = _mapping(action_spec["extension"], "invalid_action_extension", "world_action.extension")
    _exact_keys(action, WORLD_ACTION_FIELDS, "unknown_config_field", "world_action.canonical")
    _exact_keys(extension, ACTION_EXTENSION_FIELDS, "unknown_config_field", "world_action.extension")
    if validate_world_action(action):
        raise ControlledWorldContractError("invalid_world_action", "Canonical WorldAction validation failed.", blockers=validate_world_action(action))
    if action["intervention_type"] != contract["intervention_type"] or extension["modality"] != contract["modality"]:
        raise ControlledWorldContractError("family_action_mismatch", "Family action type or modality is not canonical.")
    delay = action_spec["response_delay_steps"]
    if not isinstance(delay, int) or isinstance(delay, bool) or delay < 1 or delay >= len(times):
        raise ControlledWorldContractError("invalid_action_delay", "Response delay must be inside the time grid.")
    _unit_value(extension["dose_or_intensity"], name="world_action.extension.dose_or_intensity")
    _unit_value(extension["duration"], name="world_action.extension.duration")
    semantics = _mapping(extension["action_semantics"], "invalid_action_extension", "world_action.extension.action_semantics")
    _exact_keys(semantics, ACTION_SEMANTICS_FIELDS, "unknown_config_field", "world_action.extension.action_semantics")

    support = _mapping(value["support_domain"], "invalid_support_domain", "support_domain")
    _exact_keys(
        support,
        frozenset({"support_domain_uid", "supported_intervention_types", "supported_modalities", "supported_routes", "supported_schedule_kinds", "supported_intensity", "supported_duration", "supported_information_observables", "ood_taxonomy"}),
        "unknown_config_field",
        "support_domain",
    )
    for field in ("supported_intensity", "supported_duration"):
        bounds = _mapping(support[field], "invalid_support_domain", f"support_domain.{field}")
        _exact_keys(bounds, frozenset({"minimum", "maximum", "unit"}), "unknown_config_field", f"support_domain.{field}")
        minimum = _finite(bounds["minimum"], "non_finite_value", f"{field}.minimum")
        maximum = _finite(bounds["maximum"], "non_finite_value", f"{field}.maximum")
        if maximum < minimum:
            raise ControlledWorldContractError("invalid_support_domain", "Support maximum must not be below minimum.", field=field)
    taxonomy = _mapping(support["ood_taxonomy"], "invalid_ood_taxonomy", "support_domain.ood_taxonomy")
    _exact_keys(taxonomy, frozenset({"intensity_outside_support", "duration_outside_support", "biological_context_mismatch", "combination_not_supported"}), "unknown_config_field", "support_domain.ood_taxonomy")
    output_paths = _mapping(value["output_paths"], "invalid_output_paths", "output_paths")
    _exact_keys(output_paths, frozenset({"world_instance", "replay_manifest"}), "unknown_config_field", "output_paths")

    information_specs = _sequence(value["information_actions"], "invalid_information_actions", "information_actions")
    if len(information_specs) != 2:
        raise ControlledWorldContractError("invalid_information_actions", "Each family requires exactly one partial and one failed information action.")
    statuses: set[str] = set()
    information_uids: set[str] = set()
    for raw_spec in information_specs:
        binding = bind_information_action(value, _mapping(raw_spec, "invalid_information_action", "information action"))
        statuses.add(str(binding.outcome_policy["status"]))
        if binding.information_action_uid in information_uids:
            raise ControlledWorldContractError("duplicate_information_action", "Information action identities must be unique within a family.")
        information_uids.add(binding.information_action_uid)
    if statuses != {"partial", "failed"}:
        raise ControlledWorldContractError("invalid_information_actions", "Each family requires partial and failed information outcomes.")
    default_binding = bind_world_action(value)
    default_support = classify_world_action(value, default_binding.canonical_world_action, default_binding.controlled_extension)
    if default_support.status != "supported":
        raise ControlledWorldContractError("default_action_not_supported", "The configured default action must be inside support.")
    ensure_finite_payload(value)
    ensure_no_protected_references(value)
    return value


def validate_controlled_world_config(config: Mapping[str, object]) -> dict[str, Any]:
    value = controlled_world_config(config)
    _exact_keys(
        value,
        frozenset({"schema_version", "evidence_class", "benchmark_qualified", "real_world_formal_agent_evaluation", "source_policy", "authorities", "family_manifest", "validation_audit_path", "claim_boundary", "families"}),
        "unknown_config_field",
        "controlled_neural_world",
    )
    if value["evidence_class"] != "controlled_world_non_empirical" or value["benchmark_qualified"] is not False:
        raise ControlledWorldContractError("claim_boundary_violation", "Controlled-world evidence cannot qualify the benchmark.")
    if value["real_world_formal_agent_evaluation"] is not False:
        raise ControlledWorldContractError("real_world_evaluation_forbidden", "Real-world formal Agent evaluation is closed.")
    source_policy = _mapping(value["source_policy"], "invalid_source_policy", "source_policy")
    if source_policy != {"strict_public_only_flag": True, "controlled_access_required": False}:
        raise ControlledWorldContractError("public_boundary_violation", "The public-only source boundary changed.")
    authorities = _mapping(value["authorities"], "invalid_authority", "authorities")
    _exact_keys(authorities, frozenset({"hidden_state_loader_roles", "runtime_reader_roles", "six_context_outcome_access", "protected_holdout_access"}), "unknown_config_field", "authorities")
    validate_zero_access(
        {
            "six_context_outcome_access": authorities["six_context_outcome_access"],
            "protected_holdout_access": authorities["protected_holdout_access"],
        }
    )
    if authorities["hidden_state_loader_roles"] != ["benchmark_harness"] or authorities["runtime_reader_roles"] != ["operational_twin", "agent"]:
        raise ControlledWorldContractError("oracle_authority_invalid", "Controlled-world authorities are not canonical.")
    claim = _mapping(value["claim_boundary"], "claim_boundary_violation", "claim_boundary")
    if claim != CLAIM_BOUNDARY:
        raise ControlledWorldContractError("claim_boundary_violation", "Controlled-world claim boundaries changed.")
    manifest_spec = _mapping(value["family_manifest"], "invalid_family_manifest", "family_manifest")
    _exact_keys(manifest_spec, frozenset({"schema_version", "expected_family_uids", "output_path", "benchmark_split_assignment", "sealed_test_partitions", "information_headroom_qualification", "benchmark_qualification"}), "unknown_config_field", "family_manifest")
    if manifest_spec != {
        "schema_version": FAMILY_MANIFEST_SCHEMA_VERSION,
        "expected_family_uids": list(FAMILY_UIDS),
        "output_path": "Input/processed/controlled_neural_world/controlled_world_family_manifest.json",
        "benchmark_split_assignment": "not_created",
        "sealed_test_partitions": "not_created",
        "information_headroom_qualification": "not_run",
        "benchmark_qualification": "not_issued",
    }:
        raise ControlledWorldContractError("family_manifest_contract_invalid", "Aggregate family-manifest contract changed.")
    families = [_validate_family_config(_mapping(row, "invalid_world", "family")) for row in _sequence(value["families"], "invalid_world", "families")]
    if len(families) != 3 or {str(row["family_uid"]) for row in families} != FAMILY_UID_SET:
        raise ControlledWorldContractError("unsupported_world_family", "Exactly the three contracted families are required.")
    unique_fields: dict[str, list[object]] = {
        "family_uid": [row["family_uid"] for row in families],
        "world_uid": [row["world"]["world_uid"] for row in families],
        "latent_generator_lineage_uid": [row["generator"]["latent_generator_lineage_uid"] for row in families],
        "implementation_uid": [row["generator"]["implementation_uid"] for row in families],
        "seed": [row["generator"]["seed"] for row in families],
    }
    unique_fields["action_uid"] = [row["world_action"]["canonical"]["action_uid"] for row in families]
    unique_fields["output_path"] = [
        path for row in families for path in (row["output_paths"]["world_instance"], row["output_paths"]["replay_manifest"])
    ]
    unique_fields["information_action_uid"] = [
        spec["canonical"]["information_action_uid"] for row in families for spec in row["information_actions"]
    ]
    for name, items in unique_fields.items():
        if len(items) != len(set(items)):
            raise ControlledWorldContractError("duplicate_family_identity", "Cross-family identities and paths must be unique.", field=name)
    all_subseeds = [seed for row in families for seed in row["generator"]["subseeds"].values()]
    if len(all_subseeds) != len(set(all_subseeds)):
        raise ControlledWorldContractError("duplicate_seed_identity", "Family sub-seeds must be globally unique.")
    reserved_paths = [manifest_spec["output_path"], value["validation_audit_path"]]
    if len(set(unique_fields["output_path"] + reserved_paths)) != len(unique_fields["output_path"] + reserved_paths):
        raise ControlledWorldContractError("duplicate_output_path", "Family, manifest, and audit paths must be unique.")
    ensure_finite_payload(value)
    ensure_no_protected_references(value)
    return value


def classify_world_action(
    config: Mapping[str, object],
    canonical_action: Mapping[str, object],
    extension: Mapping[str, object],
) -> SupportDecision:
    family = dict(config)
    support = _mapping(family["support_domain"], "invalid_support_domain", "support_domain")
    if canonical_action.get("intervention_type") not in support["supported_intervention_types"]:
        return SupportDecision("unsupported", ["unsupported_intervention_type"], str(support["support_domain_uid"]))
    if extension.get("modality") not in support["supported_modalities"]:
        return SupportDecision("unsupported", ["unsupported_modality"], str(support["support_domain_uid"]))
    if canonical_action.get("route") not in support["supported_routes"]:
        return SupportDecision("unsupported", ["unsupported_route"], str(support["support_domain_uid"]))
    schedule = _mapping(canonical_action.get("schedule"), "invalid_world_action", "schedule")
    if schedule.get("kind") not in support["supported_schedule_kinds"]:
        return SupportDecision("unsupported", ["unsupported_schedule"], str(support["support_domain_uid"]))
    reasons: list[str] = []
    taxonomy = support["ood_taxonomy"]
    if list(extension.get("combination", [])):
        reasons.append(str(taxonomy["combination_not_supported"]))
    context = _mapping(extension.get("biological_context"), "invalid_biological_context", "biological_context")
    if context != family["world"]["biological_context"]:
        reasons.append(str(taxonomy["biological_context_mismatch"]))
    for field, support_field, taxonomy_field in (
        ("dose_or_intensity", "supported_intensity", "intensity_outside_support"),
        ("duration", "supported_duration", "duration_outside_support"),
    ):
        value = _mapping(extension.get(field), "invalid_action_extension", field)
        bounds = _mapping(support[support_field], "invalid_support_domain", support_field)
        number = _finite(value.get("value"), "invalid_action_extension", f"{field}.value")
        if value.get("unit") != bounds.get("unit"):
            raise ControlledWorldContractError("invalid_unit", f"{field} unit does not match the support contract.", field=field)
        if number < float(bounds["minimum"]) or number > float(bounds["maximum"]):
            reasons.append(str(taxonomy[taxonomy_field]))
    return SupportDecision("out_of_distribution" if reasons else "supported", sorted(set(reasons)), str(support["support_domain_uid"]))


def bind_world_action(
    config: Mapping[str, object],
    canonical_action: Mapping[str, object] | None = None,
    extension: Mapping[str, object] | None = None,
) -> ControlledWorldActionBinding:
    family = dict(config)
    spec = _mapping(family["world_action"], "invalid_world_action", "world_action")
    configured_action = _mapping(spec["canonical"], "invalid_world_action", "registered canonical WorldAction")
    configured_extension = _mapping(spec["extension"], "invalid_action_extension", "registered controlled action extension")
    action = _mapping(
        canonical_action if canonical_action is not None else configured_action,
        "invalid_world_action",
        "canonical WorldAction",
    )
    controlled_extension = _mapping(
        extension if extension is not None else configured_extension,
        "invalid_action_extension",
        "controlled action extension",
    )
    _exact_keys(action, WORLD_ACTION_FIELDS, "unknown_world_action_field", "canonical WorldAction")
    _exact_keys(controlled_extension, ACTION_EXTENSION_FIELDS, "unknown_action_extension_field", "controlled action extension")
    blockers = validate_world_action(action)
    if blockers:
        raise ControlledWorldContractError("invalid_world_action", "Canonical WorldAction validation failed.", blockers=blockers)
    ensure_no_protected_references({"action": action, "extension": controlled_extension})
    if action["action_uid"] != configured_action["action_uid"]:
        raise ControlledWorldContractError("cross_family_action_reuse", "WorldAction identity does not belong to this family.")

    mutable_action_fields = {"dose", "duration"}
    immutable_action_mismatches = sorted(
        field
        for field in WORLD_ACTION_FIELDS - mutable_action_fields
        if action.get(field) != configured_action.get(field)
    )
    if immutable_action_mismatches:
        raise ControlledWorldContractError(
            "action_mapping_mismatch",
            "WorldAction immutable semantics must match the registered family action.",
            fields=immutable_action_mismatches,
        )

    mutable_extension_fields = {"dose_or_intensity", "duration"}
    immutable_extension_mismatches = sorted(
        field
        for field in ACTION_EXTENSION_FIELDS - mutable_extension_fields
        if controlled_extension.get(field) != configured_extension.get(field)
    )
    if immutable_extension_mismatches:
        raise ControlledWorldContractError(
            "action_mapping_mismatch",
            "Controlled action immutable semantics must match the registered family action.",
            fields=immutable_extension_mismatches,
        )

    configured_duration = _unit_value(configured_action["duration"], name="registered WorldAction duration")
    action_duration = _unit_value(action["duration"], name="WorldAction duration")
    controlled_duration = _unit_value(controlled_extension["duration"], name="controlled action duration")
    if (
        action_duration["unit"] != configured_duration["unit"]
        or controlled_duration["unit"] != configured_duration["unit"]
        or action_duration["value"] != controlled_duration["value"]
    ):
        raise ControlledWorldContractError(
            "action_mapping_mismatch",
            "Only the registered duration unit may be used and canonical/controlled duration values must agree.",
            fields=["duration"],
        )

    configured_intensity = _unit_value(configured_extension["dose_or_intensity"], name="registered dose_or_intensity")
    dose_or_intensity = _unit_value(controlled_extension["dose_or_intensity"], name="dose_or_intensity")
    if dose_or_intensity["unit"] != configured_intensity["unit"]:
        raise ControlledWorldContractError(
            "action_mapping_mismatch",
            "Only the registered dose or intensity unit may be used.",
            fields=["dose_or_intensity"],
        )

    semantics = _mapping(controlled_extension["action_semantics"], "invalid_action_extension", "controlled action semantics")
    _exact_keys(semantics, ACTION_SEMANTICS_FIELDS, "unknown_action_extension_field", "controlled action semantics")
    expected_semantics = {
        "target_set": action["target_set"],
        "agent_set": action["agent_set"],
        "schedule": action["schedule"],
        "route": action["route"],
        "support_domain_ref": action["support_domain_ref"],
        "provenance_ref": action["action_provenance_ref"],
    }
    if semantics != expected_semantics:
        raise ControlledWorldContractError("action_mapping_mismatch", "Controlled action semantics must map exactly to WorldAction.")
    if action["support_domain_ref"] != family["support_domain"]["support_domain_uid"]:
        raise ControlledWorldContractError("support_domain_mismatch", "WorldAction support domain is not canonical.")
    if controlled_extension["modality"] != action["intervention_type"]:
        raise ControlledWorldContractError("action_mapping_mismatch", "Controlled modality must preserve intervention type.")
    stimulation = _mapping(action["stimulation_or_environment"], "invalid_world_action", "stimulation_or_environment")
    if stimulation.get("modality") != controlled_extension["modality"]:
        raise ControlledWorldContractError("action_mapping_mismatch", "Controlled modality must map to the perturbation program.")
    biological_context = _mapping(controlled_extension["biological_context"], "invalid_biological_context", "controlled biological context")
    expected_scope = {"region": [biological_context.get("region")], "compartment": [biological_context.get("compartment")]}
    if action["state_scope"] != expected_scope:
        raise ControlledWorldContractError("action_mapping_mismatch", "Controlled biological context must map to state_scope.")
    if action["intervention_type"] == "pharmacological":
        if action["dose"] != dose_or_intensity["value"] or action["dose_unit"] != dose_or_intensity["unit"]:
            raise ControlledWorldContractError("action_mapping_mismatch", "Pharmacological dose and unit must map exactly.")
    elif action["dose"] is not None or action["dose_unit"] is not None:
        raise ControlledWorldContractError("action_mapping_mismatch", "Non-pharmacological controlled actions cannot carry dose fields.")
    delay = spec["response_delay_steps"]
    if not isinstance(delay, int) or isinstance(delay, bool) or delay < 1:
        raise ControlledWorldContractError("invalid_action_delay", "Response delay must be a positive integer.")
    ensure_finite_payload({"action": action, "extension": controlled_extension})
    payload = {"family_uid": family["family_uid"], "canonical_world_action": action, "controlled_extension": controlled_extension, "response_delay_steps": delay}
    return ControlledWorldActionBinding(
        binding_uid=stable_uid("controlled_world_action_binding", payload),
        canonical_world_action=action,
        controlled_extension=controlled_extension,
        response_delay_steps=delay,
        mapping_status="canonical_mapping_validated",
    )


def bind_information_action(config: Mapping[str, object], spec: Mapping[str, object]) -> ControlledInformationActionBinding:
    family = dict(config)
    _exact_keys(spec, frozenset({"canonical", "outcome_policy"}), "unknown_information_spec_field", "information action spec")
    action = _mapping(spec["canonical"], "invalid_information_action", "canonical InformationAction")
    _exact_keys(action, INFORMATION_ACTION_FIELDS, "unknown_information_action_field", "canonical InformationAction")
    blockers = validate_information_action(action)
    if blockers:
        raise ControlledWorldContractError("invalid_information_action", "Canonical InformationAction validation failed.", blockers=blockers)
    expected_action_uid = family["world_action"]["canonical"]["action_uid"]
    if action["candidate_uids"] != [expected_action_uid]:
        raise ControlledWorldContractError("cross_family_action_reuse", "InformationAction candidate must be the family WorldAction.")
    requested = {str(item) for item in action["requested_observables"]}
    if requested & HIDDEN_REQUESTS:
        code = "hidden_optimum_request_forbidden" if requested & {"hidden_optimum", "true_utility"} else "oracle_access_forbidden"
        raise ControlledWorldContractError(code, "Information actions cannot request harness-only values.")
    supported = set(family["support_domain"]["supported_information_observables"])
    if not requested.issubset(supported):
        raise ControlledWorldContractError("unsupported_information_request", "Information action requests unsupported observables.")
    if action["support_domain_ref"] != family["support_domain"]["support_domain_uid"]:
        raise ControlledWorldContractError("support_domain_mismatch", "InformationAction support domain is not canonical.")
    measurement_context = _mapping(action["measurement_context"], "invalid_information_action", "measurement_context")
    _exact_keys(measurement_context, frozenset({"region", "compartment", "time_index"}), "unknown_information_action_field", "measurement_context")
    world_context = family["world"]["biological_context"]
    if measurement_context["region"] != world_context["region"] or measurement_context["compartment"] != world_context["compartment"]:
        raise ControlledWorldContractError("information_context_mismatch", "InformationAction context must match its family.")
    time_index = measurement_context["time_index"]
    time_count = len(family["world"]["time_grid"]["values"])
    if not isinstance(time_index, int) or isinstance(time_index, bool) or time_index < 0 or time_index >= time_count:
        raise ControlledWorldContractError("invalid_observation_index", "Information measurement index is outside the time grid.")
    cost = _unit_value(action["expected_cost"], name="information_action.expected_cost")
    _unit_value(action["expected_runtime"], name="information_action.expected_runtime")
    if float(cost["value"]) <= 0:
        raise ControlledWorldContractError("invalid_information_cost", "Information actions must have positive cost.")
    outcome = _mapping(spec["outcome_policy"], "invalid_information_outcome", "information outcome policy")
    _exact_keys(outcome, frozenset({"status", "measured_observables", "missing_observables", "noise_offsets", "failure_code"}), "unknown_information_outcome_field", "information outcome policy")
    measured = {str(item) for item in _sequence(outcome["measured_observables"], "invalid_information_outcome", "measured_observables")}
    missing = {str(item) for item in _sequence(outcome["missing_observables"], "invalid_information_outcome", "missing_observables")}
    if outcome["status"] not in {"partial", "failed"} or not missing or measured & missing or measured | missing != requested:
        raise ControlledWorldContractError("invalid_information_outcome", "Information result must be partial or failed and preserve missingness.")
    offsets = _mapping(outcome["noise_offsets"], "invalid_information_outcome", "noise_offsets")
    if set(offsets) - measured:
        raise ControlledWorldContractError("invalid_information_outcome", "Noise offsets are allowed only for measured observables.")
    ensure_finite_payload(offsets)
    if outcome["status"] == "partial" and (not measured or outcome["failure_code"] is not None):
        raise ControlledWorldContractError("invalid_information_outcome", "Partial information requires measurements and no failure code.")
    if outcome["status"] == "failed" and (measured or offsets or not outcome["failure_code"]):
        raise ControlledWorldContractError("invalid_information_outcome", "Failed information requires no measurements, offsets, or imputation.")
    ensure_no_protected_references(spec)
    return ControlledInformationActionBinding(str(action["information_action_uid"]), action, outcome)


def validate_runtime_instance(
    value: Mapping[str, object],
    *,
    config: Mapping[str, object] | None = None,
    repository_root: Path | None = None,
    config_file_identity: str | None = None,
) -> dict[str, Any]:
    instance = _mapping(value, "invalid_runtime_instance", "runtime instance")
    ensure_no_runtime_secrets(instance)
    _exact_keys(instance, RUNTIME_INSTANCE_FIELDS, "unknown_runtime_field", "runtime instance")
    if instance.get("schema_version") != INSTANCE_SCHEMA_VERSION or instance.get("authority_role") != "runtime_observable":
        raise ControlledWorldContractError("runtime_authority_invalid", "Runtime instance authority is not canonical.")
    if instance.get("evidence_class") != "controlled_world_non_empirical":
        raise ControlledWorldContractError("claim_boundary_violation", "Runtime evidence class is not controlled-world evidence.")
    validate_zero_access(_mapping(instance["access_counters"], "invalid_access_counters", "access_counters"))
    if _mapping(instance["claim_boundary"], "claim_boundary_violation", "claim_boundary") != CLAIM_BOUNDARY:
        raise ControlledWorldContractError("claim_boundary_violation", "Runtime claim boundaries changed.")
    identity = _mapping(instance["world_identity"], "invalid_world_identity", "world_identity")
    _exact_keys(identity, frozenset({"world_uid", "world_family_uid", "latent_generator_lineage_uid", "biological_context"}), "unknown_world_identity_field", "world_identity")
    if identity["world_family_uid"] not in FAMILY_UID_SET:
        raise ControlledWorldContractError("unsupported_world_family", "Runtime family is outside the closed dispatch.")
    provenance = _mapping(instance["generator_provenance"], "invalid_generator_provenance", "generator_provenance")
    _exact_keys(provenance, frozenset({"implementation_uid", "implementation_version", "implementation_path", "code_identity", "config_content_identity", "config_file_identity", "seed", "seed_identity"}), "unknown_generator_provenance_field", "generator_provenance")
    for field in ("code_identity", "config_content_identity", "config_file_identity", "seed_identity"):
        if not is_sha256(provenance.get(field)):
            raise ControlledWorldContractError("invalid_replay_identity", "Runtime provenance identities must be SHA-256.", field=field)
    binding = _mapping(instance["action_binding"], "invalid_world_action", "action_binding")
    _exact_keys(binding, frozenset({"binding_uid", "canonical_world_action", "controlled_extension", "response_delay_steps", "mapping_status"}), "unknown_action_binding_field", "action_binding")
    support = _mapping(instance["support_status"], "invalid_support_status", "support_status")
    _exact_keys(support, frozenset({"status", "reason_codes", "support_domain_ref"}), "unknown_support_field", "support_status")
    observations = _sequence(instance["observations"], "invalid_shape", "observations")
    if len(observations) < 3:
        raise ControlledWorldContractError("invalid_shape", "Runtime observations require at least three timepoints.")
    for observation in observations:
        row = _mapping(observation, "invalid_shape", "observation")
        _exact_keys(row, frozenset({"time", "measurements", "missing_observables"}), "unknown_observation_field", "observation")
        time = _mapping(row["time"], "invalid_shape", "observation time")
        _exact_keys(time, frozenset({"value", "unit"}), "unknown_unit_value_field", "observation time")
        _finite(time["value"], "non_finite_value", "observation.time")
        measurements = _mapping(row["measurements"], "invalid_shape", "measurements")
        missing = _sequence(row["missing_observables"], "invalid_shape", "missing_observables")
        if set(measurements) & set(missing):
            raise ControlledWorldContractError("invalid_shape", "Measured and missing observables must be disjoint.")
        for variable, raw_measurement in measurements.items():
            measurement = _mapping(raw_measurement, "invalid_shape", str(variable))
            _exact_keys(measurement, frozenset({"value", "unit"}), "unknown_unit_value_field", str(variable))
            _finite(measurement["value"], "non_finite_value", str(variable))
    results = _sequence(instance["information_results"], "invalid_information_outcome", "information_results")
    if len(results) != 2:
        raise ControlledWorldContractError("invalid_information_outcome", "Runtime requires exactly two information results.")
    for raw_result in results:
        result = _mapping(raw_result, "invalid_information_outcome", "information result")
        _exact_keys(result, frozenset({"information_action_uid", "status", "cost", "measurements", "missing_observables", "failure_code", "support_status", "provenance_ref"}), "unknown_information_result_field", "information result")
        cost = _unit_value(result["cost"], name="information result cost")
        if float(cost["value"]) <= 0:
            raise ControlledWorldContractError("invalid_information_cost", "Information result cost must be positive.")
        measurements = _mapping(result["measurements"], "invalid_information_outcome", "information measurements")
        for variable, raw_measurement in measurements.items():
            measurement = _mapping(raw_measurement, "invalid_information_outcome", str(variable))
            _exact_keys(measurement, frozenset({"value", "unit"}), "unknown_unit_value_field", str(variable))
            _finite(measurement["value"], "non_finite_value", str(variable))
        missing = _sequence(result["missing_observables"], "invalid_information_outcome", "missing_observables")
        if not missing or len(missing) != len(set(missing)):
            raise ControlledWorldContractError("invalid_information_outcome", "Missing observables must be unique and non-empty.")
        if result["status"] == "partial" and (not measurements or result["failure_code"] is not None):
            raise ControlledWorldContractError("invalid_information_outcome", "Partial information result is contradictory.")
        if result["status"] == "failed" and (measurements or not result["failure_code"]):
            raise ControlledWorldContractError("invalid_information_outcome", "Failed information result is contradictory.")
        if result["status"] not in {"partial", "failed"}:
            raise ControlledWorldContractError("invalid_information_outcome", "Information result status is unsupported.")
    total = _unit_value(instance["information_cost_total"], name="information_cost_total")
    expected_total = sum(float(_mapping(result["cost"], "invalid_information_cost", "cost")["value"]) for result in results)
    if total["unit"] != "cost_unit" or float(total["value"]) != expected_total:
        raise ControlledWorldContractError("information_result_config_mismatch", "Information cost total is not authoritative.")
    if config is not None:
        source = validate_controlled_world_config(config)
        family_uid = str(identity["world_family_uid"])
        family = select_family_config(source, family_uid)
        from Libs.utils.data.controlled_world_runtime import _generate_runtime_instance_unvalidated, sha256_file  # noqa: PLC0415

        resolved_root = Path(repository_root).resolve() if repository_root is not None else Path(__file__).resolve().parents[3]
        source_path = resolved_root / "Libs/configs/data/controlled_neural_world.yaml"
        resolved_identity = config_file_identity or sha256_file(source_path)
        expected, _ = _generate_runtime_instance_unvalidated(
            source,
            family_uid=family_uid,
            repository_root=resolved_root,
            config_file_identity=resolved_identity,
        )
        if instance != expected:
            mismatches = sorted(field for field in set(instance) | set(expected) if instance.get(field) != expected.get(field))
            raise ControlledWorldContractError(
                "runtime_config_mismatch",
                "Runtime instance does not exactly match the authoritative config-derived instance.",
                fields=mismatches,
                expected_identity=canonical_sha256(expected),
                observed_identity=canonical_sha256(instance),
            )
        action_binding = bind_world_action(family, binding["canonical_world_action"], binding["controlled_extension"])
        if action_binding.to_dict() != binding:
            raise ControlledWorldContractError("runtime_config_mismatch", "Runtime action binding is not authoritative.", fields=["action_binding"])
    return instance


def validate_replay_manifest(value: Mapping[str, object]) -> dict[str, Any]:
    manifest = _mapping(value, "invalid_replay_manifest", "replay manifest")
    ensure_no_runtime_secrets(manifest)
    _exact_keys(manifest, REPLAY_MANIFEST_FIELDS, "unknown_replay_manifest_field", "replay manifest")
    if manifest.get("schema_version") != REPLAY_SCHEMA_VERSION:
        raise ControlledWorldContractError("replay_schema_mismatch", "Replay manifest schema is not canonical.")
    if manifest.get("world_family_uid") not in FAMILY_UID_SET:
        raise ControlledWorldContractError("invalid_replay_identity", "Replay family is outside the closed dispatch.")
    for field in ("replay_uid", "world_uid", "latent_generator_lineage_uid", "generator_implementation_uid", "generator_implementation_version"):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            raise ControlledWorldContractError("invalid_replay_identity", "Replay identity fields must be explicit.", field=field)
    if not isinstance(manifest.get("seed"), int) or isinstance(manifest.get("seed"), bool) or manifest["seed"] < 0:
        raise ControlledWorldContractError("invalid_seed", "Replay seed must be non-negative.")
    for field in ("generator_code_identity", "config_content_identity", "config_file_identity", "seed_identity", "action_sequence_identity", "information_sequence_identity", "observable_output_identity"):
        if not is_sha256(manifest.get(field)):
            raise ControlledWorldContractError("invalid_replay_identity", "Replay identity must be SHA-256.", field=field)
    for lineage_name, expected_count in (("input_lineage", 3), ("output_lineage", 1)):
        rows = _sequence(manifest[lineage_name], "invalid_replay_lineage", lineage_name)
        if len(rows) != expected_count:
            raise ControlledWorldContractError("invalid_replay_lineage", "Replay lineage count is not canonical.", field=lineage_name)
        for index, raw_row in enumerate(rows):
            row = _mapping(raw_row, "invalid_replay_lineage", f"{lineage_name}[{index}]")
            _exact_keys(row, frozenset({"role", "artifact_ref", "content_identity"}), "unknown_replay_lineage_field", f"{lineage_name}[{index}]")
            if not is_sha256(row["content_identity"]):
                raise ControlledWorldContractError("invalid_replay_identity", "Replay lineage identity must be SHA-256.")
    validate_zero_access(_mapping(manifest["access_counters"], "invalid_access_counters", "access_counters"))
    reproduction = _mapping(manifest["deterministic_reproduction"], "invalid_replay_contract", "deterministic_reproduction")
    if reproduction != {"required": True, "byte_equivalent_canonical_json": True, "replay_count": 1, "status": "replay_identity_frozen"}:
        raise ControlledWorldContractError("invalid_replay_contract", "Deterministic replay requirements changed.")
    return manifest


def authoritative_family_manifest_row(
    family: Mapping[str, object],
    instance: Mapping[str, object],
    replay: Mapping[str, object],
    *,
    config_file_identity: str,
) -> dict[str, object]:
    family_value = dict(family)
    family_uid = str(family_value["family_uid"])
    instance_value = validate_runtime_instance(instance)
    replay_value = validate_replay_manifest(replay)
    identity = _mapping(instance_value["world_identity"], "invalid_world_identity", "world_identity")
    provenance = _mapping(
        instance_value["generator_provenance"],
        "invalid_generator_provenance",
        "generator_provenance",
    )
    world = _mapping(family_value["world"], "invalid_world", "world")
    generator = _mapping(family_value["generator"], "invalid_generator", "generator")
    output_paths = _mapping(family_value["output_paths"], "invalid_output_paths", "output_paths")
    runtime_content_identity = canonical_sha256(instance_value)
    replay_content_identity = canonical_sha256(replay_value)
    expected_seed_identity = canonical_sha256(
        {
            "family_uid": family_uid,
            "world_uid": world["world_uid"],
            "seed": generator["seed"],
            "seed_policy": generator["seed_policy"],
            "subseeds": generator["subseeds"],
        }
    )
    expected_action_identity = action_sequence_identity(
        _mapping(instance_value["action_binding"], "invalid_world_action", "action_binding")
    )
    expected_information_identity = information_sequence_identity(
        [bind_information_action(family_value, dict(spec)).to_dict() for spec in family_value["information_actions"]]
    )
    expected_replay_uid = stable_uid(
        "controlled_world_replay",
        {
            "world_uid": identity["world_uid"],
            "generator_code_identity": provenance["code_identity"],
            "config_content_identity": provenance["config_content_identity"],
            "seed_identity": provenance["seed_identity"],
            "action_sequence_identity": expected_action_identity,
            "information_sequence_identity": expected_information_identity,
            "observable_output_identity": runtime_content_identity,
        },
    )
    expected_input_lineage = [
        {
            "role": "generator_implementation",
            "artifact_ref": provenance["implementation_path"],
            "content_identity": provenance["code_identity"],
        },
        {
            "role": "controlled_world_config",
            "artifact_ref": "Libs/configs/data/controlled_neural_world.yaml",
            "content_identity": config_file_identity,
        },
        {
            "role": "family_seed_policy",
            "artifact_ref": f"controlled_neural_world.families.{family_uid}.generator.seed",
            "content_identity": expected_seed_identity,
        },
    ]
    expected_output_lineage = [
        {
            "role": "runtime_observable",
            "artifact_ref": output_paths["world_instance"],
            "content_identity": runtime_content_identity,
        }
    ]
    package_mismatches: list[str] = []
    equality_checks = {
        "world_identity.world_uid": identity.get("world_uid") == world.get("world_uid"),
        "world_identity.world_family_uid": identity.get("world_family_uid") == family_uid,
        "world_identity.latent_generator_lineage_uid": identity.get("latent_generator_lineage_uid")
        == generator.get("latent_generator_lineage_uid"),
        "world_identity.biological_context": identity.get("biological_context") == world.get("biological_context"),
        "generator_provenance.implementation_uid": provenance.get("implementation_uid")
        == generator.get("implementation_uid"),
        "generator_provenance.implementation_version": provenance.get("implementation_version")
        == generator.get("implementation_version"),
        "generator_provenance.implementation_path": provenance.get("implementation_path")
        == generator.get("implementation_path"),
        "generator_provenance.seed": provenance.get("seed") == generator.get("seed"),
        "generator_provenance.seed_identity": provenance.get("seed_identity") == expected_seed_identity,
        "replay.world_family_uid": replay_value.get("world_family_uid") == family_uid,
        "replay.world_uid": replay_value.get("world_uid") == identity.get("world_uid"),
        "replay.latent_generator_lineage_uid": replay_value.get("latent_generator_lineage_uid")
        == identity.get("latent_generator_lineage_uid"),
        "replay.generator_implementation_uid": replay_value.get("generator_implementation_uid")
        == provenance.get("implementation_uid"),
        "replay.generator_implementation_version": replay_value.get("generator_implementation_version")
        == provenance.get("implementation_version"),
        "replay.generator_code_identity": replay_value.get("generator_code_identity")
        == provenance.get("code_identity"),
        "replay.config_content_identity": replay_value.get("config_content_identity")
        == provenance.get("config_content_identity")
        == canonical_sha256(family_value),
        "replay.config_file_identity": replay_value.get("config_file_identity")
        == provenance.get("config_file_identity")
        == config_file_identity,
        "replay.seed": replay_value.get("seed") == provenance.get("seed"),
        "replay.seed_identity": replay_value.get("seed_identity") == provenance.get("seed_identity"),
        "replay.replay_uid": replay_value.get("replay_uid") == expected_replay_uid,
        "replay.action_sequence_identity": replay_value.get("action_sequence_identity") == expected_action_identity,
        "replay.information_sequence_identity": replay_value.get("information_sequence_identity")
        == expected_information_identity,
        "replay.observable_output_identity": replay_value.get("observable_output_identity")
        == runtime_content_identity,
        "replay.input_lineage": replay_value.get("input_lineage") == expected_input_lineage,
        "replay.output_lineage": replay_value.get("output_lineage") == expected_output_lineage,
        "replay.access_counters": replay_value.get("access_counters") == instance_value.get("access_counters"),
    }
    package_mismatches.extend(field for field, matches in equality_checks.items() if not matches)
    if package_mismatches:
        raise ControlledWorldContractError(
            "family_package_lineage_mismatch",
            "Runtime and replay package lineage do not agree with their family authority.",
            family_uid=family_uid,
            fields=sorted(package_mismatches),
        )
    return {
        "family_uid": family_uid,
        "world_uid": identity["world_uid"],
        "latent_generator_lineage_uid": identity["latent_generator_lineage_uid"],
        "seed": provenance["seed"],
        "seed_identity": provenance["seed_identity"],
        "runtime_artifact_ref": output_paths["world_instance"],
        "runtime_content_identity": runtime_content_identity,
        "replay_artifact_ref": output_paths["replay_manifest"],
        "replay_content_identity": replay_content_identity,
        "replay_uid": replay_value["replay_uid"],
    }


def validate_family_manifest(
    value: Mapping[str, object],
    *,
    config: Mapping[str, object],
    config_file_identity: str,
    authoritative_packages: Mapping[str, tuple[Mapping[str, object], Mapping[str, object]]],
) -> dict[str, Any]:
    manifest = _mapping(value, "invalid_family_manifest", "family manifest")
    ensure_no_runtime_secrets(manifest)
    _exact_keys(manifest, FAMILY_MANIFEST_FIELDS, "unknown_family_manifest_field", "family manifest")
    if manifest.get("schema_version") != FAMILY_MANIFEST_SCHEMA_VERSION:
        raise ControlledWorldContractError("family_manifest_schema_mismatch", "Family manifest schema is not canonical.")
    if manifest.get("evidence_class") != "controlled_world_non_empirical" or manifest.get("real_world_formal_agent_evaluation") is not False:
        raise ControlledWorldContractError("claim_boundary_violation", "Family manifest evidence boundary changed.")
    if manifest.get("family_count") != 3 or manifest.get("family_uids") != list(FAMILY_UIDS):
        raise ControlledWorldContractError("family_manifest_incomplete", "Family manifest must enumerate exactly the three families.")
    if manifest.get("claim_boundary") != CLAIM_BOUNDARY:
        raise ControlledWorldContractError("claim_boundary_violation", "Family manifest claim boundary changed.")
    for field, expected in (
        ("benchmark_split_assignment", "not_created"),
        ("sealed_test_partitions", "not_created"),
        ("information_headroom_qualification", "not_run"),
        ("benchmark_qualification", "not_issued"),
    ):
        if manifest.get(field) != expected:
            raise ControlledWorldContractError("benchmark_scope_violation", "Family manifest cannot freeze or qualify the benchmark.", field=field)
    validate_zero_access(_mapping(manifest["access_counters"], "invalid_access_counters", "access_counters"))
    if not is_sha256(manifest.get("source_config_identity")):
        raise ControlledWorldContractError("invalid_family_manifest", "Source config identity must be SHA-256.")
    rows = _sequence(manifest["families"], "invalid_family_manifest", "families")
    if len(rows) != 3:
        raise ControlledWorldContractError("family_manifest_incomplete", "Family manifest row count is not canonical.")
    normalized: list[dict[str, object]] = []
    for raw_row in rows:
        row = _mapping(raw_row, "invalid_family_manifest", "family row")
        _exact_keys(row, frozenset(FAMILY_MANIFEST_ROW_FIELDS), "unknown_family_manifest_field", "family row")
        if row["family_uid"] not in FAMILY_UID_SET:
            raise ControlledWorldContractError("unsupported_world_family", "Manifest family is outside the closed dispatch.")
        for field in ("seed_identity", "runtime_content_identity", "replay_content_identity"):
            if not is_sha256(row[field]):
                raise ControlledWorldContractError("invalid_family_manifest", "Manifest identities must be SHA-256.", field=field)
        normalized.append(row)
    for field in ("family_uid", "world_uid", "latent_generator_lineage_uid", "seed", "seed_identity", "runtime_artifact_ref", "replay_artifact_ref", "replay_uid"):
        items = [row[field] for row in normalized]
        if len(items) != len(set(items)):
            raise ControlledWorldContractError("duplicate_family_identity", "Family manifest identities must be unique.", field=field)
    if [row["family_uid"] for row in normalized] != list(FAMILY_UIDS):
        raise ControlledWorldContractError("family_manifest_order_invalid", "Family manifest order must be canonical.")
    expected_manifest_uid = stable_uid(
        "controlled_world_family_manifest",
        {"source_config_identity": manifest["source_config_identity"], "families": normalized},
    )
    if manifest.get("manifest_uid") != expected_manifest_uid:
        raise ControlledWorldContractError("family_manifest_identity_mismatch", "Family manifest identity is not content-derived.")
    if manifest["source_config_identity"] != config_file_identity:
        raise ControlledWorldContractError("family_manifest_config_mismatch", "Family manifest source config identity changed.")
    if manifest["source_config_ref"] != "Libs/configs/data/controlled_neural_world.yaml":
        raise ControlledWorldContractError(
            "family_manifest_config_mismatch",
            "Family manifest source config reference changed.",
            field="source_config_ref",
        )
    source = validate_controlled_world_config(config)
    if set(authoritative_packages) != set(FAMILY_UIDS):
        raise ControlledWorldContractError(
            "family_manifest_authority_incomplete",
            "Authoritative runtime/replay packages must cover exactly the three families.",
        )
    for row in normalized:
        family_uid = str(row["family_uid"])
        family = select_family_config(source, family_uid)
        instance, replay = authoritative_packages[family_uid]
        expected_row = authoritative_family_manifest_row(
            family,
            instance,
            replay,
            config_file_identity=config_file_identity,
        )
        mismatches = [field for field in FAMILY_MANIFEST_ROW_FIELDS if row.get(field) != expected_row.get(field)]
        if mismatches:
            raise ControlledWorldContractError(
                "family_manifest_authority_mismatch",
                "Family manifest row differs from the referenced runtime/replay authorities.",
                family_uid=family_uid,
                fields=mismatches,
            )
    return manifest


def action_sequence_identity(binding: Mapping[str, object]) -> str:
    return canonical_sha256(
        {
            "canonical_world_action": binding["canonical_world_action"],
            "controlled_extension": binding["controlled_extension"],
            "response_delay_steps": binding["response_delay_steps"],
        }
    )


def information_sequence_identity(bindings: Sequence[Mapping[str, object]]) -> str:
    return canonical_sha256(list(bindings))
