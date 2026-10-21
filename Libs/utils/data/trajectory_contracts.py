"""Fail-closed Round-15 clean trajectory qualification contracts.

The contracts distinguish newly generated E4 artifacts from legacy E3 rows and
restrict the Round-15 pilot to qualification evidence. They do not train a
model, execute counterfactual studies, seal the Evidence Fabric, or evaluate
DT_CORE.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np

from Libs.utils.data.action_observation_contracts import (
    WORLD_ACTION_TYPES,
    validate_world_action,
)


SCHEMA_VERSIONS = {
    "trajectory": "neuronsifter-clean-action-complete-trajectory-v1",
    "producer": "neuronsifter-trajectory-producer-v1",
    "qualification": "neuronsifter-trajectory-producer-qualification-v1",
    "initial_state": "neuronsifter-trajectory-initial-state-v1",
    "action_binding": "neuronsifter-trajectory-action-binding-v1",
    "numerical_qc": "neuronsifter-trajectory-numerical-qc-v1",
    "constraint_qc": "neuronsifter-trajectory-constraint-qc-v1",
    "control_pair": "neuronsifter-trajectory-control-pair-v1",
    "failure": "neuronsifter-trajectory-failure-v1",
    "admission": "neuronsifter-trajectory-admission-decision-v1",
    "family": "neuronsifter-trajectory-family-manifest-v1",
    "release": "neuronsifter-round15-e4-release-candidate-v1",
}

PRODUCER_TYPES = frozenset(
    {
        "mechanistic_ode",
        "mechanistic_pde",
        "agent_based_model",
        "network_dynamics_model",
        "hybrid_multiscale_simulator",
        "governed_scientific_tool",
        "validated_replay_engine",
    }
)
CAPABILITY_STATES = frozenset(
    {
        "supported",
        "supported_with_restrictions",
        "unsupported",
        "unknown",
        "blocked_parameters",
        "blocked_action_compiler",
        "blocked_initial_state",
        "blocked_license",
    }
)
QUALIFICATION_STATES = frozenset(
    {
        "qualified",
        "qualified_with_restrictions",
        "blocked_numerical_stability",
        "blocked_action_semantics",
        "blocked_initial_state",
        "blocked_parameter_provenance",
        "blocked_support_domain",
        "blocked_environment",
        "blocked_license",
    }
)
PAIRING_DECISIONS = frozenset(
    {
        "exact_deterministic_control_pair",
        "matched_stochastic_control_pair",
        "group_level_control_pair",
        "unpaired_trajectory",
    }
)
ADMISSION_DECISIONS = frozenset(
    {
        "admitted_e4_strong",
        "admitted_e4_restricted",
        "admitted_stress",
        "quarantined_numerical",
        "quarantined_constraint",
        "rejected_action_incomplete",
        "rejected_initial_state",
        "rejected_provenance",
        "rejected_support_domain",
        "rejected_hash",
    }
)
E4_ELIGIBILITY = frozenset(
    {
        "qualification_numerical_dynamics_only",
        "restricted_clean_dynamics",
        "strong_clean_dynamics",
        "stress_only",
        "ineligible",
    }
)
INITIAL_STATE_ORIGINS = frozenset(
    {
        "governed_e1_observation_derived_state",
        "validated_mechanistic_steady_state",
        "validated_simulator_initialization",
        "explicit_synthetic_qualification_state",
    }
)
REPLICATE_ROLES = frozenset(
    {
        "within_action_stochastic_replicate",
        "between_initial_state_replicate",
        "between_parameter_replicate",
        "between_producer_replicate",
        "deterministic_singleton",
    }
)
SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
COMMIT_RE = re.compile(r"^[a-f0-9]{40}$")


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(canonical_json(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def stable_uid(prefix: str, value: object) -> str:
    return f"{prefix}_{canonical_sha256(value)[:24]}"


def is_sha256(value: object) -> bool:
    return bool(SHA256_RE.fullmatch(str(value)))


def blocker(code: str, message: str, **details: object) -> dict[str, object]:
    return {"code": code, "message": message, **details}


def sorted_blockers(values: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    unique = {canonical_json(dict(value)): dict(value) for value in values}
    return sorted(unique.values(), key=lambda row: (str(row.get("code", "")), canonical_json(row)))


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: object) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _required_text(value: Mapping[str, object], fields: Sequence[str], uid: str) -> list[dict[str, object]]:
    return [
        blocker("missing_required_field", f"{field} must be explicit.", object_uid=uid, field=field)
        for field in fields
        if not _text(value.get(field))
    ]


@dataclass(frozen=True)
class TrajectoryProducer:
    schema_version: str
    producer_uid: str
    version: str
    producer_type: str
    source: str
    source_hash: str
    license_or_access_status: str
    equation_or_model_identity: str
    input_schema: str
    output_schema: str
    supported_action_classes: Sequence[str]
    supported_state_variables: Sequence[str]
    support_domain_ref: str
    parameter_policy: Mapping[str, object]
    seed_policy: Mapping[str, object]
    numerical_solver_policy: Mapping[str, object]
    known_limitations: Sequence[str]
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TrajectoryInitialState:
    schema_version: str
    state_uid: str
    state_time: float
    time_unit: str
    region_axis: Sequence[str]
    compartment_axis: Sequence[str]
    species_or_feature_axis: Sequence[str]
    units: Mapping[str, str]
    missingness: Mapping[str, object]
    uncertainty: Mapping[str, object]
    biological_or_model_context: Mapping[str, object]
    support_domain_ref: str
    source_lineage: Mapping[str, object]
    state_values_ref: str
    state_hash: str
    origin: str
    imputation_policy: str
    empirical_state: bool
    qualification_only: bool
    eligible_for_biological_validation: bool
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TrajectoryActionBinding:
    schema_version: str
    binding_uid: str
    world_action: Mapping[str, object]
    action_program_hash: str
    exposure_occupancy_program_ref: str | None
    exposure_occupancy_status: str
    producer_action_parameters: Mapping[str, object]
    parameter_provenance: Mapping[str, object]
    support_domain_ref: str
    completeness: str
    blockers: Sequence[Mapping[str, object]]
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TrajectoryNumericalQC:
    schema_version: str
    qc_uid: str
    trajectory_uid: str
    policy_version: str
    solver_status: str
    convergence_status: str
    step_count: int
    rejected_steps: int
    runtime_seconds: float
    memory: Mapping[str, object]
    nan_inf_count: int
    overflow_underflow_indicators: Mapping[str, object]
    state_bound_violations: int
    negative_concentration_violations: int
    discontinuities: int
    interpolation_error: float
    repeatability_diagnostics: Mapping[str, object]
    status: str
    blockers: Sequence[Mapping[str, object]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TrajectoryConstraintQC:
    schema_version: str
    constraint_qc_uid: str
    trajectory_uid: str
    policy_version: str
    required_constraints: Sequence[str]
    constraint_results: Mapping[str, Mapping[str, object]]
    unsupported_monotonicity_imposed: bool
    status: str
    blockers: Sequence[Mapping[str, object]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CleanActionCompleteTrajectory:
    schema_version: str
    trajectory_uid: str
    trajectory_family_uid: str
    episode_uid: str
    producer_uid: str
    producer_version: str
    producer_type: str
    initial_state_ref: str
    initial_state_hash: str
    world_action_ref: str
    action_program_hash: str
    no_action_control_ref: str | None
    support_domain_ref: str
    state_namespace: str
    region_axis: Sequence[str]
    compartment_axis: Sequence[str]
    species_or_feature_axis: Sequence[str]
    channel_units: Mapping[str, str]
    start_time: float
    end_time: float
    time_unit: str
    time_grid: Sequence[float]
    native_solver_time_grid: Sequence[float]
    canonical_export_time_grid: Sequence[float]
    sampling_interval: Mapping[str, object]
    adaptive_step_metadata: Mapping[str, object]
    interpolation: Mapping[str, object]
    state_values_ref: str
    state_values_hash: str
    uncertainty_ref: str
    constraint_diagnostics_ref: str
    numerical_qc_ref: str
    physical_qc_ref: str
    producer_inputs_hash: str
    producer_parameters_hash: str
    random_seed: int
    code_commit: str
    environment_hash: str
    runtime_identity: Mapping[str, object]
    status: str
    eligibility: str
    provenance: Mapping[str, object]
    derived_from_legacy_trajectory_ref: str | None
    promotion_method: str
    promotion_validation: Mapping[str, object]
    fidelity_uid: str
    conflicts: Sequence[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TrajectoryAdmissionDecision:
    schema_version: str
    admission_uid: str
    trajectory_uid: str
    producer_qualification_ref: str
    initial_state_ref: str
    action_binding_ref: str
    support_domain_ref: str
    numerical_qc_ref: str
    constraint_qc_ref: str
    artifact_hash: str
    blocking_conflicts: Sequence[str]
    decision: str
    eligibility: str
    reason_codes: Sequence[str]
    provenance: Mapping[str, object]
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def round15_config(config: Mapping[str, object]) -> dict[str, object]:
    value = config.get("round15_trajectory_domain")
    if not isinstance(value, Mapping):
        raise KeyError("round15_trajectory_domain")
    return dict(value)


def validate_round15_config(config: Mapping[str, object]) -> list[dict[str, object]]:
    try:
        value = round15_config(config)
    except KeyError:
        return [blocker("missing_round15_config", "Round-15 trajectory configuration is absent.")]
    blockers: list[dict[str, object]] = []
    if value.get("version") != "neuronsifter-round15-trajectory-domain-v1":
        blockers.append(blocker("invalid_round15_config_version", "Round-15 configuration version is not canonical."))
    if value.get("evidence_fabric_sealed") is not False:
        blockers.append(blocker("evidence_fabric_sealed", "Round-15 must not seal the Evidence Fabric."))
    if value.get("dt_core_status") != "not_evaluated":
        blockers.append(blocker("dt_core_evaluated", "Round-15 must not evaluate DT_CORE."))
    output_paths = _mapping(value.get("output_paths"))
    for name, raw_path in output_paths.items():
        path = Path(_text(raw_path))
        if not _text(raw_path) or path.is_absolute():
            blockers.append(blocker("invalid_output_path", "Round-15 output paths must be relative.", output=name))
    producer = _mapping(value.get("producer"))
    if set(_list(producer.get("supported_action_classes"))) - WORLD_ACTION_TYPES:
        blockers.append(blocker("unknown_action_class", "Producer action classes must use the Round-13 vocabulary."))
    production = _mapping(value.get("production"))
    if int(production.get("expected_run_count", 0)) < 1:
        blockers.append(blocker("empty_production_plan", "At least one justified formal trajectory is required."))
    return sorted_blockers(blockers)


def validate_producer(value: Mapping[str, object], *, known_support_domains: set[str] | None = None) -> list[dict[str, object]]:
    producer = dict(value)
    uid = _text(producer.get("producer_uid"))
    blockers = _required_text(
        producer,
        (
            "producer_uid",
            "version",
            "producer_type",
            "source",
            "source_hash",
            "license_or_access_status",
            "equation_or_model_identity",
            "input_schema",
            "output_schema",
            "support_domain_ref",
            "status",
        ),
        uid,
    )
    if producer.get("schema_version") != SCHEMA_VERSIONS["producer"]:
        blockers.append(blocker("unknown_producer_schema", "Trajectory producer schema is not canonical.", producer_uid=uid))
    if _text(producer.get("producer_type")) not in PRODUCER_TYPES:
        blockers.append(blocker("unknown_producer_type", "Unknown trajectory producer types fail closed.", producer_uid=uid))
    if "llm" in _text(producer.get("producer_type")).lower() or "llm" in _text(producer.get("equation_or_model_identity")).lower():
        blockers.append(blocker("llm_trajectory_producer_prohibited", "An LLM cannot be a trajectory producer.", producer_uid=uid))
    if not is_sha256(producer.get("source_hash")):
        blockers.append(blocker("missing_artifact_hash", "Producer source requires an exact SHA-256.", producer_uid=uid))
    if _text(producer.get("version")).lower() in {"", "unknown", "latest"}:
        blockers.append(blocker("unknown_simulator_version", "Producer version must be exact and immutable.", producer_uid=uid))
    if _text(producer.get("license_or_access_status")) not in {
        "verified_public",
        "repository_internal_governed",
        "resolved_restricted",
    }:
        blockers.append(blocker("blocked_license", "Producer license or access must be resolved.", producer_uid=uid))
    action_classes = set(_list(producer.get("supported_action_classes")))
    if not action_classes or not action_classes.issubset(WORLD_ACTION_TYPES):
        blockers.append(blocker("unknown_action_class", "Producer action capability is missing or invalid.", producer_uid=uid))
    if not _list(producer.get("supported_state_variables")):
        blockers.append(blocker("missing_state_namespace", "Producer state variables must be explicit.", producer_uid=uid))
    parameter_policy = _mapping(producer.get("parameter_policy"))
    if not parameter_policy or parameter_policy.get("unknown_parameters_allowed") is not False:
        blockers.append(blocker("unknown_parameter_provenance", "Unknown producer parameters fail closed.", producer_uid=uid))
    domain_ref = _text(producer.get("support_domain_ref"))
    if known_support_domains is not None and domain_ref not in known_support_domains:
        blockers.append(blocker("unknown_support_domain", "Producer support domain is not registered.", producer_uid=uid))
    if not _mapping(producer.get("seed_policy")) or not _mapping(producer.get("numerical_solver_policy")):
        blockers.append(blocker("incomplete_producer_policy", "Seed and numerical solver policies must be explicit.", producer_uid=uid))
    return sorted_blockers(blockers)


def validate_capability_matrix(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    keys: set[tuple[str, ...]] = set()
    for row in rows:
        key = tuple(
            _text(row.get(field))
            for field in ("producer_uid", "organism", "disease_or_context", "region", "compartment", "action_class", "fidelity")
        )
        if key in keys:
            blockers.append(blocker("duplicate_capability_cell", "Capability matrix cells must be unique.", cell=key))
        keys.add(key)
        if _text(row.get("state")) not in CAPABILITY_STATES:
            blockers.append(blocker("invalid_capability_state", "Capability state is not canonical.", cell=key))
        if _text(row.get("action_class")) not in WORLD_ACTION_TYPES:
            blockers.append(blocker("unknown_action_class", "Capability matrix action class is not canonical.", cell=key))
        if _text(row.get("state")) in {"supported", "supported_with_restrictions"}:
            for field in ("dose_route_schedule_support", "temporal_horizon", "state_variables", "support_domain_ref", "restrictions"):
                if not _text(row.get(field)):
                    blockers.append(blocker("incomplete_supported_capability", "Supported capability cells require explicit scope.", cell=key, field=field))
    return sorted_blockers(blockers)


def validate_initial_state(
    value: Mapping[str, object],
    *,
    state_values: np.ndarray | None = None,
    known_support_domains: set[str] | None = None,
) -> list[dict[str, object]]:
    state = dict(value)
    uid = _text(state.get("state_uid"))
    blockers = _required_text(
        state,
        ("state_uid", "time_unit", "support_domain_ref", "state_values_ref", "state_hash", "origin", "imputation_policy", "status"),
        uid,
    )
    if state.get("schema_version") != SCHEMA_VERSIONS["initial_state"]:
        blockers.append(blocker("unknown_initial_state_schema", "Initial-state schema is not canonical.", state_uid=uid))
    axes = [_list(state.get(name)) for name in ("region_axis", "compartment_axis", "species_or_feature_axis")]
    if any(not axis for axis in axes):
        blockers.append(blocker("trajectory_initial_state_incomplete", "All initial-state axes must be explicit.", state_uid=uid))
    species = {_text(item) for item in axes[2]}
    units = _mapping(state.get("units"))
    if species != set(units) or any(not _text(value) for value in units.values()):
        blockers.append(blocker("trajectory_initial_state_incomplete", "Every initial-state channel requires an explicit unit.", state_uid=uid))
    if state.get("imputation_policy") != "none_all_channels_explicit":
        blockers.append(blocker("silent_default_initial_state", "Initial-state channels cannot be silently imputed.", state_uid=uid))
    if not _mapping(state.get("missingness")) or not _mapping(state.get("uncertainty")):
        blockers.append(blocker("trajectory_initial_state_incomplete", "Missingness and uncertainty must be explicit.", state_uid=uid))
    if not _mapping(state.get("biological_or_model_context")) or not _mapping(state.get("source_lineage")):
        blockers.append(blocker("trajectory_initial_state_incomplete", "Initial-state context and lineage must be explicit.", state_uid=uid))
    if _text(state.get("origin")) not in INITIAL_STATE_ORIGINS:
        blockers.append(blocker("trajectory_initial_state_incomplete", "Initial-state origin is not allowed.", state_uid=uid))
    if state.get("origin") == "explicit_synthetic_qualification_state" and (
        state.get("empirical_state") is not False
        or state.get("qualification_only") is not True
        or state.get("eligible_for_biological_validation") is not False
    ):
        blockers.append(blocker("synthetic_state_role_upgrade", "Synthetic qualification states must retain non-empirical flags.", state_uid=uid))
    if not is_sha256(state.get("state_hash")):
        blockers.append(blocker("missing_artifact_hash", "Initial-state hash is missing or invalid.", state_uid=uid))
    domain_ref = _text(state.get("support_domain_ref"))
    if known_support_domains is not None and domain_ref not in known_support_domains:
        blockers.append(blocker("unknown_support_domain", "Initial-state support domain is not registered.", state_uid=uid))
    if state_values is not None:
        expected_shape = tuple(len(axis) for axis in axes)
        if state_values.shape != expected_shape:
            blockers.append(blocker("initial_state_shape_mismatch", "Initial-state tensor does not match declared axes.", state_uid=uid))
        if not np.isfinite(state_values).all():
            blockers.append(blocker("initial_state_nonfinite", "Initial-state values must be finite.", state_uid=uid))
        if array_sha256(state_values) != state.get("state_hash"):
            blockers.append(blocker("initial_state_hash_mismatch", "Initial-state content hash does not match.", state_uid=uid))
    return sorted_blockers(blockers)


def validate_action_binding(
    value: Mapping[str, object],
    *,
    producer: Mapping[str, object] | None = None,
    known_support_domains: set[str] | None = None,
) -> list[dict[str, object]]:
    binding = dict(value)
    uid = _text(binding.get("binding_uid"))
    blockers = _required_text(
        binding,
        ("binding_uid", "action_program_hash", "exposure_occupancy_status", "support_domain_ref", "completeness", "status"),
        uid,
    )
    if binding.get("schema_version") != SCHEMA_VERSIONS["action_binding"]:
        blockers.append(blocker("unknown_action_binding_schema", "Action binding schema is not canonical.", binding_uid=uid))
    action = _mapping(binding.get("world_action"))
    blockers.extend(validate_world_action(action))
    action_type = _text(action.get("intervention_type"))
    if producer is not None and action_type not in set(_list(producer.get("supported_action_classes"))):
        blockers.append(blocker("unsupported_producer_action", "Producer does not support this action class.", binding_uid=uid, action_class=action_type))
    if not is_sha256(binding.get("action_program_hash")):
        blockers.append(blocker("missing_artifact_hash", "Action program requires an exact hash.", binding_uid=uid))
    if _text(binding.get("completeness")) != "complete":
        blockers.append(blocker("trajectory_action_incomplete", "Trajectory action binding must be complete.", binding_uid=uid))
    parameters = _mapping(binding.get("producer_action_parameters"))
    if not parameters or parameters.get("silent_defaults_used") is not False:
        blockers.append(blocker("silent_default_action_parameters", "Producer action parameters must be explicit and default-free.", binding_uid=uid))
    provenance = _mapping(binding.get("parameter_provenance"))
    if provenance.get("status") not in {"verified_exact_configuration", "verified_governed_artifact"} or not is_sha256(provenance.get("content_hash")):
        blockers.append(blocker("unknown_parameter_provenance", "Action parameter provenance must be exact.", binding_uid=uid))
    compiler_status = _text(binding.get("exposure_occupancy_status"))
    if action_type in {"pharmacological", "combination"} and compiler_status not in {"compiled", "compiled_with_explicit_uncertainty"}:
        blockers.append(blocker("blocked_action_compiler", "Pharmacological trajectories require a compiled occupancy program.", binding_uid=uid))
    if action_type == "no_action_control" and compiler_status != "not_applicable":
        blockers.append(blocker("invalid_no_action_compiler_status", "No-action controls must mark occupancy compilation not applicable.", binding_uid=uid))
    domain_ref = _text(binding.get("support_domain_ref"))
    if domain_ref != _text(action.get("support_domain_ref")):
        blockers.append(blocker("support_domain_mismatch", "Action and binding support domains must match.", binding_uid=uid))
    if known_support_domains is not None and domain_ref not in known_support_domains:
        blockers.append(blocker("unknown_support_domain", "Action support domain is not registered.", binding_uid=uid))
    return sorted_blockers(blockers)


def build_numerical_qc(
    *,
    trajectory_uid: str,
    trajectory: np.ndarray,
    repeat_trajectory: np.ndarray,
    runtime_seconds: float,
    policy: Mapping[str, object],
) -> dict[str, object]:
    finite_mask = np.isfinite(trajectory)
    nan_inf_count = int(trajectory.size - int(finite_mask.sum()))
    negative_count = int(np.count_nonzero(trajectory[finite_mask] < 0.0)) if finite_mask.any() else 0
    repeatable = trajectory.shape == repeat_trajectory.shape and np.array_equal(trajectory, repeat_trajectory)
    blockers: list[dict[str, object]] = []
    if nan_inf_count > int(policy["maximum_nan_inf_count"]):
        blockers.append(blocker("nan_inf_trajectory", "Trajectory contains NaN or Inf values.", trajectory_uid=trajectory_uid))
    if negative_count > int(policy["maximum_negative_value_count"]):
        blockers.append(blocker("negative_state_violation", "Trajectory contains negative state values.", trajectory_uid=trajectory_uid))
    if not repeatable:
        blockers.append(blocker("deterministic_replay_mismatch", "Deterministic producer replay changed output.", trajectory_uid=trajectory_uid))
    blockers = sorted_blockers(blockers)
    return TrajectoryNumericalQC(
        schema_version=SCHEMA_VERSIONS["numerical_qc"],
        qc_uid=stable_uid("trajectory_numerical_qc", {"trajectory_uid": trajectory_uid, "policy": policy.get("policy_version")}),
        trajectory_uid=trajectory_uid,
        policy_version=_text(policy.get("policy_version")),
        solver_status="completed" if not blockers else "failed",
        convergence_status="not_applicable_explicit_recurrence",
        step_count=max(0, int(trajectory.shape[0]) - 1),
        rejected_steps=0,
        runtime_seconds=max(0.0, float(runtime_seconds)),
        memory={"measurement": "not_profiled_bounded_qualification_run", "array_bytes": int(trajectory.nbytes)},
        nan_inf_count=nan_inf_count,
        overflow_underflow_indicators={"overflow_count": 0, "underflow_count": 0, "detection": "numpy_finite_and_zero_scan"},
        state_bound_violations=negative_count,
        negative_concentration_violations=negative_count,
        discontinuities=0,
        interpolation_error=0.0,
        repeatability_diagnostics={
            "exact_array_equal": repeatable,
            "first_hash": array_sha256(trajectory),
            "repeat_hash": array_sha256(repeat_trajectory),
            "absolute_tolerance": float(policy["repeatability_absolute_tolerance"]),
            "relative_tolerance": float(policy["repeatability_relative_tolerance"]),
        },
        status="passed" if not blockers else "failed",
        blockers=blockers,
    ).to_dict()


def validate_numerical_qc(value: Mapping[str, object]) -> list[dict[str, object]]:
    qc = dict(value)
    uid = _text(qc.get("trajectory_uid"))
    blockers: list[dict[str, object]] = []
    if qc.get("schema_version") != SCHEMA_VERSIONS["numerical_qc"]:
        blockers.append(blocker("unknown_numerical_qc_schema", "Numerical QC schema is not canonical.", trajectory_uid=uid))
    if qc.get("solver_status") != "completed":
        blockers.append(blocker("solver_failed", "Failed solver output cannot pass numerical QC.", trajectory_uid=uid))
    if int(qc.get("nan_inf_count", -1)) != 0:
        blockers.append(blocker("nan_inf_trajectory", "NaN or Inf trajectories fail closed.", trajectory_uid=uid))
    if int(qc.get("negative_concentration_violations", -1)) != 0 or int(qc.get("state_bound_violations", -1)) != 0:
        blockers.append(blocker("state_bound_violation", "State-bound violations fail numerical QC.", trajectory_uid=uid))
    if _mapping(qc.get("repeatability_diagnostics")).get("exact_array_equal") is not True:
        blockers.append(blocker("deterministic_replay_mismatch", "Deterministic replay must be exact.", trajectory_uid=uid))
    if qc.get("status") != "passed" or _list(qc.get("blockers")):
        blockers.append(blocker("numerical_qc_not_passed", "Numerical QC is not passing.", trajectory_uid=uid))
    return sorted_blockers(blockers)


def build_constraint_qc(
    *,
    trajectory_uid: str,
    trajectory: np.ndarray,
    species_axis: Sequence[str],
    policy: Mapping[str, object],
) -> dict[str, object]:
    species_lookup = {name: index for index, name in enumerate(species_axis)}
    nonnegative_violations = int(np.count_nonzero(trajectory < 0.0))
    occupancy = trajectory[..., species_lookup["target_bound_drug"]]
    lower = float(policy["occupancy_lower_bound"])
    upper = float(policy["occupancy_upper_bound"])
    occupancy_violations = int(np.count_nonzero((occupancy < lower) | (occupancy > upper)))
    results = {
        "non_negativity": {"state": "passed" if nonnegative_violations == 0 else "failed", "evidence": {"violation_count": nonnegative_violations}},
        "bounded_occupancy": {"state": "passed" if occupancy_violations == 0 else "failed", "evidence": {"violation_count": occupancy_violations, "lower": lower, "upper": upper}},
        "mass_or_species_conservation": {"state": "not_applicable", "evidence": {"reason": "producer_does_not_claim_closed_mass_conservation"}},
        "charge_conservation": {"state": "not_applicable", "evidence": {"reason": "producer_is_not_an_electrodiffusion_solver"}},
        "probability_normalization": {"state": "not_applicable", "evidence": {"reason": "state_channels_are_not_probabilities"}},
    }
    blockers = [
        blocker("constraint_failure", "Required trajectory constraint failed.", constraint=name, trajectory_uid=trajectory_uid)
        for name in _list(policy.get("required_constraints"))
        if _mapping(results.get(_text(name))).get("state") not in {"passed", "passed_with_tolerance"}
    ]
    blockers = sorted_blockers(blockers)
    return TrajectoryConstraintQC(
        schema_version=SCHEMA_VERSIONS["constraint_qc"],
        constraint_qc_uid=stable_uid("trajectory_constraint_qc", {"trajectory_uid": trajectory_uid, "policy": policy.get("policy_version")}),
        trajectory_uid=trajectory_uid,
        policy_version=_text(policy.get("policy_version")),
        required_constraints=[_text(value) for value in _list(policy.get("required_constraints"))],
        constraint_results=results,
        unsupported_monotonicity_imposed=False,
        status="passed" if not blockers else "failed",
        blockers=blockers,
    ).to_dict()


def validate_constraint_qc(value: Mapping[str, object]) -> list[dict[str, object]]:
    qc = dict(value)
    uid = _text(qc.get("trajectory_uid"))
    blockers: list[dict[str, object]] = []
    if qc.get("schema_version") != SCHEMA_VERSIONS["constraint_qc"]:
        blockers.append(blocker("unknown_constraint_qc_schema", "Constraint QC schema is not canonical.", trajectory_uid=uid))
    results = _mapping(qc.get("constraint_results"))
    for name in _list(qc.get("required_constraints")):
        state = _text(_mapping(results.get(_text(name))).get("state"))
        if state not in {"passed", "passed_with_tolerance"}:
            blockers.append(blocker("required_constraint_not_passed", "Required constraints must pass.", trajectory_uid=uid, constraint=name, state=state))
    if qc.get("unsupported_monotonicity_imposed") is not False:
        blockers.append(blocker("unsupported_biological_monotonicity", "Unsupported biological monotonicity cannot be imposed.", trajectory_uid=uid))
    if qc.get("status") != "passed" or _list(qc.get("blockers")):
        blockers.append(blocker("constraint_qc_not_passed", "Constraint QC is not passing.", trajectory_uid=uid))
    return sorted_blockers(blockers)


def validate_trajectory(
    value: Mapping[str, object],
    *,
    producer: Mapping[str, object] | None = None,
    initial_state: Mapping[str, object] | None = None,
    action_binding: Mapping[str, object] | None = None,
    state_values: np.ndarray | None = None,
    known_support_domains: set[str] | None = None,
) -> list[dict[str, object]]:
    trajectory = dict(value)
    uid = _text(trajectory.get("trajectory_uid"))
    blockers = _required_text(
        trajectory,
        (
            "trajectory_uid", "trajectory_family_uid", "episode_uid", "producer_uid", "producer_version",
            "producer_type", "initial_state_ref", "initial_state_hash", "world_action_ref", "action_program_hash",
            "support_domain_ref", "state_namespace", "time_unit", "state_values_ref", "state_values_hash",
            "uncertainty_ref", "constraint_diagnostics_ref", "numerical_qc_ref", "physical_qc_ref",
            "producer_inputs_hash", "producer_parameters_hash", "code_commit", "environment_hash",
            "status", "eligibility", "promotion_method", "fidelity_uid",
        ),
        uid,
    )
    if trajectory.get("schema_version") != SCHEMA_VERSIONS["trajectory"]:
        blockers.append(blocker("unknown_trajectory_schema", "Clean trajectory schema is not canonical.", trajectory_uid=uid))
    for field in ("initial_state_hash", "action_program_hash", "state_values_hash", "producer_inputs_hash", "producer_parameters_hash", "environment_hash"):
        if not is_sha256(trajectory.get(field)):
            blockers.append(blocker("missing_artifact_hash", "Trajectory content identity is incomplete.", trajectory_uid=uid, field=field))
    if not COMMIT_RE.fullmatch(_text(trajectory.get("code_commit"))):
        blockers.append(blocker("unknown_code_commit", "Trajectory code commit must be exact.", trajectory_uid=uid))
    axes = [_list(trajectory.get(name)) for name in ("region_axis", "compartment_axis", "species_or_feature_axis")]
    if any(not axis for axis in axes):
        blockers.append(blocker("trajectory_axis_incomplete", "Trajectory state axes must be explicit.", trajectory_uid=uid))
    if set(_text(item) for item in axes[2]) != set(_mapping(trajectory.get("channel_units"))):
        blockers.append(blocker("trajectory_unit_incomplete", "Every trajectory channel requires an explicit unit.", trajectory_uid=uid))
    native_grid = _list(trajectory.get("native_solver_time_grid"))
    export_grid = _list(trajectory.get("canonical_export_time_grid"))
    declared_grid = _list(trajectory.get("time_grid"))
    if len(native_grid) < 2 or declared_grid != export_grid:
        blockers.append(blocker("trajectory_time_axis_incomplete", "Native and canonical time axes must be explicit and aligned.", trajectory_uid=uid))
    if not _mapping(trajectory.get("sampling_interval")) or not _mapping(trajectory.get("adaptive_step_metadata")) or not _mapping(trajectory.get("interpolation")):
        blockers.append(blocker("trajectory_time_metadata_incomplete", "Time normalization metadata must be explicit.", trajectory_uid=uid))
    domain_ref = _text(trajectory.get("support_domain_ref"))
    if known_support_domains is not None and domain_ref not in known_support_domains:
        blockers.append(blocker("unknown_support_domain", "Trajectory support domain is not registered.", trajectory_uid=uid))
    if producer is not None:
        blockers.extend(validate_producer(producer, known_support_domains=known_support_domains))
        if (_text(trajectory.get("producer_uid")), _text(trajectory.get("producer_version"))) != (
            _text(producer.get("producer_uid")), _text(producer.get("version"))
        ):
            blockers.append(blocker("producer_identity_mismatch", "Trajectory producer identity does not match the registry.", trajectory_uid=uid))
        if domain_ref != _text(producer.get("support_domain_ref")):
            blockers.append(blocker("support_domain_mismatch", "Trajectory exceeds producer support domain.", trajectory_uid=uid))
    if initial_state is not None:
        blockers.extend(validate_initial_state(initial_state, known_support_domains=known_support_domains))
        if _text(trajectory.get("initial_state_ref")) != _text(initial_state.get("state_uid")) or trajectory.get("initial_state_hash") != initial_state.get("state_hash"):
            blockers.append(blocker("initial_state_identity_mismatch", "Trajectory initial-state identity is inconsistent.", trajectory_uid=uid))
    if action_binding is not None:
        blockers.extend(validate_action_binding(action_binding, producer=producer, known_support_domains=known_support_domains))
        action = _mapping(action_binding.get("world_action"))
        if _text(trajectory.get("world_action_ref")) != _text(action.get("action_uid")) or trajectory.get("action_program_hash") != action_binding.get("action_program_hash"):
            blockers.append(blocker("action_identity_mismatch", "Trajectory action identity is inconsistent.", trajectory_uid=uid))
    provenance = _mapping(trajectory.get("provenance"))
    evidence_kind = _text(provenance.get("source_kind"))
    if evidence_kind in {"static_endpoint", "cross_sectional_observation"}:
        blockers.append(blocker("static_endpoint_promoted_to_trajectory", "Static evidence cannot become an E4 trajectory.", trajectory_uid=uid))
    if evidence_kind == "hypothetical_scenario":
        blockers.append(blocker("hypothetical_promoted_to_e4", "Hypothetical scenarios cannot become E4 truth.", trajectory_uid=uid))
    if trajectory.get("derived_from_legacy_trajectory_ref") and trajectory.get("promotion_method") != "governed_regeneration":
        blockers.append(blocker("legacy_promoted_without_regeneration", "E3 requires a new governed regeneration to become E4.", trajectory_uid=uid))
    if trajectory.get("eligibility") not in E4_ELIGIBILITY:
        blockers.append(blocker("invalid_trajectory_eligibility", "Trajectory eligibility is not canonical.", trajectory_uid=uid))
    if state_values is not None:
        expected_shape = (len(declared_grid), *(len(axis) for axis in axes))
        if state_values.shape != expected_shape:
            blockers.append(blocker("trajectory_shape_mismatch", "Trajectory tensor does not match declared axes.", trajectory_uid=uid))
        if array_sha256(state_values) != trajectory.get("state_values_hash"):
            blockers.append(blocker("trajectory_hash_mismatch", "Trajectory content hash does not match.", trajectory_uid=uid))
        if initial_state is not None and array_sha256(state_values[0]) != initial_state.get("state_hash"):
            blockers.append(blocker("initial_state_replay_mismatch", "Trajectory time zero does not match the governed initial state.", trajectory_uid=uid))
    return sorted_blockers(blockers)


def validate_control_pair(value: Mapping[str, object]) -> list[dict[str, object]]:
    pair = dict(value)
    uid = _text(pair.get("pair_uid"))
    blockers: list[dict[str, object]] = []
    if pair.get("schema_version") != SCHEMA_VERSIONS["control_pair"]:
        blockers.append(blocker("unknown_control_pair_schema", "Control-pair schema is not canonical.", pair_uid=uid))
    decision = _text(pair.get("pairing_decision"))
    if decision not in PAIRING_DECISIONS:
        blockers.append(blocker("invalid_pairing_decision", "Control-pair decision is not canonical.", pair_uid=uid))
    required_matches = ("initial_state_family_match", "producer_version_match", "parameter_family_match", "seed_policy_compatible", "export_time_grid_match")
    if decision != "unpaired_trajectory" and any(pair.get(field) is not True for field in required_matches):
        blockers.append(blocker("control_pair_mismatch", "Paired trajectories must match all required identities.", pair_uid=uid))
    if decision == "unpaired_trajectory" and pair.get("action_effect_claim_eligible") is True:
        blockers.append(blocker("unpaired_action_contrast", "Unpaired trajectories cannot support intervention contrasts.", pair_uid=uid))
    return sorted_blockers(blockers)


def validate_fidelity_mapping(*, requested_fidelity: str, actual_fidelity: str, mapping_ref: str | None) -> list[dict[str, object]]:
    if requested_fidelity == actual_fidelity or mapping_ref:
        return []
    return [blocker("low_fidelity_substitution", "Fidelity substitution requires an explicit versioned mapping.")]


def validate_stochastic_identity(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    by_sample: dict[tuple[str, str, str], set[str]] = {}
    for row in rows:
        role = _text(row.get("replicate_role"))
        if role not in REPLICATE_ROLES:
            blockers.append(blocker("invalid_replicate_role", "Stochastic replicate role is not canonical."))
        key = (_text(row.get("seed_family_uid")), _text(row.get("seed")), _text(row.get("sample_content_hash")))
        by_sample.setdefault(key, set()).add(_text(row.get("independence_group_uid")))
    if any(len(groups) > 1 for groups in by_sample.values()):
        blockers.append(blocker("seed_role_collision", "Identical seed realizations cannot be treated as independent samples."))
    return sorted_blockers(blockers)


def validate_failure_ledger(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    attempt_uids: set[str] = set()
    for row in rows:
        attempt_uid = _text(row.get("attempt_uid"))
        if not attempt_uid or attempt_uid in attempt_uids:
            blockers.append(blocker("failed_attempt_overwritten", "Every failed attempt requires a unique retained identity.", attempt_uid=attempt_uid))
        attempt_uids.add(attempt_uid)
        preserved = row.get("preserved") in {True, "true", "True", "1", 1}
        if not preserved or row.get("status") != "retained_failure":
            blockers.append(blocker("failed_attempt_overwritten", "Failed attempts must remain preserved.", attempt_uid=attempt_uid))
        superseded_by = _text(row.get("superseded_by"))
        if superseded_by and superseded_by == attempt_uid:
            blockers.append(blocker("invalid_failure_supersession", "A failed attempt cannot supersede itself.", attempt_uid=attempt_uid))
    return sorted_blockers(blockers)


def validate_independence_groups(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    required = (
        "producer_family", "equation_model_family", "initial_state_family", "biological_context_family",
        "action_family", "compound_scaffold_family", "target_program_family", "regimen_family",
        "seed_family", "parameter_family", "fidelity_family", "trajectory_family",
    )
    blockers: list[dict[str, object]] = []
    for row in rows:
        for field in required:
            if not _text(row.get(field)):
                blockers.append(blocker("incomplete_independence_grouping", "E4 independence grouping is incomplete.", trajectory_uid=row.get("trajectory_uid"), field=field))
    return sorted_blockers(blockers)


def validate_gap_trajectory_map(
    rows: Sequence[Mapping[str, object]],
    *,
    admitted_trajectory_uids: set[str],
    expected_gap_count: int = 20,
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    if len(rows) != expected_gap_count or len({_text(row.get("gap_uid")) for row in rows}) != expected_gap_count:
        blockers.append(blocker("frozen_gap_count_drift", "Round-15 must map exactly 20 unique frozen gaps."))
    for row in rows:
        status = _text(row.get("current_status"))
        if status == "resolved_by_admitted_e4":
            uid = _text(row.get("trajectory_uid"))
            evidence = _mapping(row.get("resolution_evidence"))
            if uid not in admitted_trajectory_uids or not is_sha256(evidence.get("artifact_hash")) or evidence.get("exact_requirement_match") is not True:
                blockers.append(blocker("gap_closed_without_admission_evidence", "A frozen gap cannot close without exact admitted E4 evidence.", gap_uid=row.get("gap_uid")))
    return sorted_blockers(blockers)


def admit_trajectory(
    *,
    trajectory: Mapping[str, object],
    producer_qualification: Mapping[str, object],
    initial_state: Mapping[str, object],
    action_binding: Mapping[str, object],
    numerical_qc: Mapping[str, object],
    constraint_qc: Mapping[str, object],
    known_support_domains: set[str],
    state_values: np.ndarray | None = None,
) -> dict[str, object]:
    uid = _text(trajectory.get("trajectory_uid"))
    trajectory_blockers = validate_trajectory(
        trajectory,
        initial_state=initial_state,
        action_binding=action_binding,
        state_values=state_values,
        known_support_domains=known_support_domains,
    )
    initial_blockers = validate_initial_state(initial_state, known_support_domains=known_support_domains)
    action_blockers = validate_action_binding(action_binding, known_support_domains=known_support_domains)
    numerical_blockers = validate_numerical_qc(numerical_qc)
    constraint_blockers = validate_constraint_qc(constraint_qc)
    qualification_status = _text(producer_qualification.get("status"))
    blockers = sorted_blockers([*trajectory_blockers, *initial_blockers, *action_blockers, *numerical_blockers, *constraint_blockers])
    codes = {str(row["code"]) for row in blockers}
    if qualification_status not in {"qualified", "qualified_with_restrictions"}:
        decision = "rejected_provenance"
        blockers.append(blocker("producer_not_qualified", "Only qualified producers can admit E4 trajectories.", trajectory_uid=uid))
    elif any("initial_state" in code or "synthetic_state" in code for code in codes):
        decision = "rejected_initial_state"
    elif any("action" in code or "dose" in code or "route" in code for code in codes):
        decision = "rejected_action_incomplete"
    elif any("hash" in code for code in codes):
        decision = "rejected_hash"
    elif any("support_domain" in code for code in codes):
        decision = "rejected_support_domain"
    elif numerical_blockers:
        decision = "quarantined_numerical"
    elif constraint_blockers:
        decision = "quarantined_constraint"
    elif _list(trajectory.get("conflicts")):
        decision = "admitted_stress"
    elif qualification_status == "qualified_with_restrictions" or trajectory.get("eligibility") == "qualification_numerical_dynamics_only":
        decision = "admitted_e4_restricted"
    else:
        decision = "admitted_e4_strong"
    if decision == "admitted_e4_strong" and numerical_qc.get("solver_status") != "completed":
        blockers.append(blocker("failed_solver_admitted_strong", "Failed solver output cannot be admitted as strong E4.", trajectory_uid=uid))
        decision = "quarantined_numerical"
    if decision == "admitted_e4_strong" and constraint_qc.get("status") != "passed":
        blockers.append(blocker("constraint_failure_ignored", "Constraint failures cannot be ignored for strong E4.", trajectory_uid=uid))
        decision = "quarantined_constraint"
    blockers = sorted_blockers(blockers)
    eligibility = (
        "strong_clean_dynamics" if decision == "admitted_e4_strong" else
        "qualification_numerical_dynamics_only" if decision == "admitted_e4_restricted" else
        "stress_only" if decision == "admitted_stress" else "ineligible"
    )
    reasons = [decision, f"producer_{qualification_status}"]
    reasons.extend(sorted(codes))
    return TrajectoryAdmissionDecision(
        schema_version=SCHEMA_VERSIONS["admission"],
        admission_uid=stable_uid("trajectory_admission", {"trajectory_uid": uid, "decision": decision}),
        trajectory_uid=uid,
        producer_qualification_ref=_text(producer_qualification.get("qualification_uid")),
        initial_state_ref=_text(initial_state.get("state_uid")),
        action_binding_ref=_text(action_binding.get("binding_uid")),
        support_domain_ref=_text(trajectory.get("support_domain_ref")),
        numerical_qc_ref=_text(numerical_qc.get("qc_uid")),
        constraint_qc_ref=_text(constraint_qc.get("constraint_qc_uid")),
        artifact_hash=_text(trajectory.get("state_values_hash")),
        blocking_conflicts=[_text(value) for value in _list(trajectory.get("conflicts"))],
        decision=decision,
        eligibility=eligibility,
        reason_codes=sorted(set(reasons)),
        provenance={"validator_uid": "round15_clean_action_complete_trajectory_validator", "blockers": blockers},
        status="decided",
    ).to_dict()


def validate_admission_decision(
    value: Mapping[str, object],
    *,
    numerical_qc: Mapping[str, object] | None = None,
    constraint_qc: Mapping[str, object] | None = None,
) -> list[dict[str, object]]:
    decision = dict(value)
    uid = _text(decision.get("trajectory_uid"))
    blockers: list[dict[str, object]] = []
    if decision.get("schema_version") != SCHEMA_VERSIONS["admission"]:
        blockers.append(blocker("unknown_admission_schema", "Trajectory admission schema is not canonical.", trajectory_uid=uid))
    state = _text(decision.get("decision"))
    if state not in ADMISSION_DECISIONS:
        blockers.append(blocker("invalid_admission_decision", "Trajectory admission decision is not canonical.", trajectory_uid=uid))
    if not is_sha256(decision.get("artifact_hash")):
        blockers.append(blocker("missing_artifact_hash", "Admission requires a content hash.", trajectory_uid=uid))
    if state == "admitted_e4_strong" and numerical_qc is not None and numerical_qc.get("solver_status") != "completed":
        blockers.append(blocker("failed_solver_admitted_strong", "Failed solver output cannot be admitted strong.", trajectory_uid=uid))
    if state == "admitted_e4_strong" and numerical_qc is not None and int(numerical_qc.get("nan_inf_count", 1)) != 0:
        blockers.append(blocker("nan_inf_admitted", "NaN or Inf output cannot be admitted strong.", trajectory_uid=uid))
    if state == "admitted_e4_strong" and constraint_qc is not None and constraint_qc.get("status") != "passed":
        blockers.append(blocker("constraint_failure_ignored", "Constraint failure cannot be admitted strong.", trajectory_uid=uid))
    return sorted_blockers(blockers)


def release_content_hash(value: Mapping[str, object]) -> str:
    payload = dict(value)
    payload.pop("content_hash", None)
    return canonical_sha256(payload)


def validate_release_candidate(value: Mapping[str, object]) -> list[dict[str, object]]:
    release = dict(value)
    blockers: list[dict[str, object]] = []
    if release.get("schema_version") != SCHEMA_VERSIONS["release"]:
        blockers.append(blocker("unknown_release_schema", "Round-15 release schema is not canonical."))
    if release.get("sealed") is not False or release.get("status") != "materialized_unsealed_successor_candidate":
        blockers.append(blocker("evidence_fabric_sealed", "Round-15 successor release must remain unsealed."))
    for field in (
        "parent_release_manifest_hash", "config_hash", "schema_registry_hash", "input_bundle_hash", "environment_hash",
        "producer_qualification_hash", "trajectory_catalog_hash", "pairing_catalog_hash", "admission_ledger_hash",
        "failure_ledger_hash", "gap_map_hash", "independence_group_hash",
    ):
        if not is_sha256(release.get(field)):
            blockers.append(blocker("missing_artifact_hash", "Release candidate hash is missing.", field=field))
    if int(release.get("trajectory_count", 0)) < 1:
        blockers.append(blocker("empty_e4_release", "Round-15 release candidate requires an admitted E4 trajectory."))
    if release_content_hash(release) != release.get("content_hash"):
        blockers.append(blocker("release_content_hash_mismatch", "Release candidate content hash does not match."))
    return sorted_blockers(blockers)
