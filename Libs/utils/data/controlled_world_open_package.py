"""Open-only paired action package construction for controlled neural worlds."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from Libs.utils.data.controlled_world_contracts import (
    FAMILY_UIDS,
    ControlledWorldContractError,
    action_sequence_identity,
    bind_world_action,
    classify_world_action,
    information_sequence_identity,
    select_family_config,
    validate_controlled_world_config,
)
from Libs.utils.data.controlled_world_runtime import generate_runtime_instance, sha256_file
from Libs.utils.data.raw_data_registry import write_json, write_text_atomic
from Libs.utils.data.trajectory_contracts import canonical_sha256, stable_uid

OPEN_PACKAGE_CONFIG_SCHEMA_VERSION = "neuronsifter-controlled-world-open-action-config-v1"
OPEN_MANIFEST_SCHEMA_VERSION = "neuronsifter-controlled-world-open-action-manifest-v1"
OPEN_PACKAGE_RECEIPT_SCHEMA_VERSION = "neuronsifter-controlled-world-open-package-receipt-v1"
OPEN_PACKAGE_POSITIVE_TERMINAL = "OPEN_ACTION_IDENTIFICATION_PACKAGE_CONSTRUCTED"
OPEN_PACKAGE_NEGATIVE_TERMINAL = "OPEN_ACTION_IDENTIFICATION_PACKAGE_NOT_CONSTRUCTED"
OPEN_ROLES = ("training", "development", "calibration")
ZERO_ACCESS_COUNTERS = {
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
CLOSED_DOWNSTREAM_FLAGS = {
    "training_authorized": False,
    "sufficient_training": False,
    "twin_qualification_pass": False,
    "formal_qualification": False,
    "agent_work_open": False,
    "test_manifest_open": False,
    "tool_gate_open": False,
    "mcp_gate_open": False,
    "agent_trace_gate_open": False,
    "memory_gate_open": False,
    "coevolution_gate_open": False,
}
FORBIDDEN_PATH_TOKENS = (
    "test_manifest",
    "sealed_partition",
    "twin_qualification",
    "agent_decision",
    "component_mechanism",
    "coevolution_robustness",
    "access_receipt",
    "access_receipts",
    "protected_holdout",
    "six_context",
    "six-context",
)


def open_package_config(config: Mapping[str, object]) -> dict[str, Any]:
    value = (
        config.get("controlled_world_open_action_identification")
        if "controlled_world_open_action_identification" in config
        else config
    )
    if not isinstance(value, Mapping):
        raise ControlledWorldContractError("missing_open_action_config", "Open action-identification config is missing.")
    return deepcopy(dict(value))


def reject_forbidden_path_tokens(path_value: str | Path, *, label: str) -> None:
    text = Path(path_value).as_posix().lower()
    if any(token in text for token in FORBIDDEN_PATH_TOKENS):
        raise ControlledWorldContractError(
            "forbidden_path_token",
            "Forbidden sealed, protected, or unopened receipt path token was rejected before content access.",
            label=label,
            path=str(path_value),
        )


def resolve_declared_output(artifact_root: Path, relative_path: object, *, label: str) -> Path:
    path = Path(str(relative_path))
    if path.is_absolute() or ".." in path.parts:
        raise ControlledWorldContractError("invalid_output_path", "Open-package output paths must be repository-relative.", label=label)
    if label in {"training_manifest", "development_manifest", "calibration_manifest", "package_receipt"}:
        reject_forbidden_path_tokens(path, label=label)
    return artifact_root / path


def validate_open_package_config(config: Mapping[str, object]) -> dict[str, Any]:
    value = open_package_config(config)
    required = {
        "schema_version",
        "evidence_class",
        "authority_role",
        "family_uids",
        "roles",
        "latent_worlds_per_family",
        "role_seed_namespaces",
        "pairing_policy",
        "action_programs",
        "development_support_cases",
        "replay_policy",
        "failure_policy",
        "access_policy",
        "closed_downstream_flags",
        "terminal_policy",
        "historical_benchmark_provenance",
        "output_paths",
    }
    if set(value) != required:
        raise ControlledWorldContractError("invalid_open_action_config", "Open action config keys are not the governed contract.", fields=sorted(set(value) ^ required))
    if value["schema_version"] != OPEN_PACKAGE_CONFIG_SCHEMA_VERSION or value["evidence_class"] != "controlled_world_non_empirical":
        raise ControlledWorldContractError("invalid_open_action_config", "Open action config schema or evidence class is invalid.")
    if value["authority_role"] != "open_action_identification_constructor":
        raise ControlledWorldContractError("invalid_open_action_authority", "Open package authority role is not canonical.")
    if tuple(value["family_uids"]) != FAMILY_UIDS or tuple(value["roles"]) != OPEN_ROLES:
        raise ControlledWorldContractError("unsupported_world_family", "Open package must use the canonical three families and open roles.")
    if dict(value["access_policy"]) != ZERO_ACCESS_COUNTERS:
        raise ControlledWorldContractError("protected_access_nonzero", "Open package access counters must remain exactly zero.")
    if dict(value["closed_downstream_flags"]) != CLOSED_DOWNSTREAM_FLAGS:
        raise ControlledWorldContractError("downstream_gate_open", "Training and downstream gates must remain closed.")
    terminal = dict(value["terminal_policy"])
    if terminal != {
        "positive_terminal": OPEN_PACKAGE_POSITIVE_TERMINAL,
        "negative_terminal": OPEN_PACKAGE_NEGATIVE_TERMINAL,
        "open_data_qualification": "pending",
    }:
        raise ControlledWorldContractError("invalid_terminal_policy", "Open package terminal policy is not exact.")
    paths = dict(value["output_paths"])
    required_paths = {
        "training_manifest",
        "development_manifest",
        "calibration_manifest",
        "package_receipt",
        "construction_audit",
        "construction_report",
        "job_receipt",
    }
    if set(paths) != required_paths:
        raise ControlledWorldContractError("invalid_output_path", "Open package output path set is incomplete.", fields=sorted(set(paths) ^ required_paths))
    banned_name_tokens = ("m1", "m2", "m3", "m4", "phase", "stage")
    for key, raw_path in paths.items():
        path = Path(str(raw_path))
        if path.is_absolute() or ".." in path.parts:
            raise ControlledWorldContractError("invalid_output_path", "Open-package output paths must be repository-relative.", field=key)
        lowered = path.as_posix().lower()
        if any(token in lowered for token in banned_name_tokens):
            raise ControlledWorldContractError("nonsemantic_artifact_name", "Open package artifact names must be semantic.", field=key)
        if key in {"training_manifest", "development_manifest", "calibration_manifest", "package_receipt"}:
            reject_forbidden_path_tokens(path, label=key)
    policy = dict(value["pairing_policy"])
    if policy.get("pair_key_field") != "counterfactual_pair_uid" or int(policy.get("supported_members_per_pair", 0)) != 2:
        raise ControlledWorldContractError("invalid_pairing_policy", "Open package requires exactly two supported members per pair.")
    latent_counts = dict(value["latent_worlds_per_family"])
    seed_namespaces = dict(value["role_seed_namespaces"])
    for role in OPEN_ROLES:
        if int(latent_counts.get(role, 0)) < 1:
            raise ControlledWorldContractError("invalid_role_cardinality", "Every open role needs at least one latent world per family.", role=role)
        namespace = dict(seed_namespaces.get(role, {}))
        if set(namespace) != {"namespace_uid", "base_seed"} or int(namespace["base_seed"]) < 0:
            raise ControlledWorldContractError("invalid_seed_namespace", "Role seed namespaces must be explicit and non-negative.", role=role)
    programs = dict(value["action_programs"])
    if set(programs) != set(FAMILY_UIDS):
        raise ControlledWorldContractError("missing_action_program", "Each family requires exactly two declared action programs.")
    for family_uid, raw_programs in programs.items():
        family_programs = list(raw_programs)
        if len(family_programs) != 2:
            raise ControlledWorldContractError("invalid_action_program_count", "Each family requires exactly two action programs.", family_uid=family_uid)
        seen = set()
        for program in family_programs:
            row = dict(program)
            if set(row) != {"action_program_uid", "dose_or_intensity", "duration"}:
                raise ControlledWorldContractError("invalid_action_program", "Action program keys are not the governed contract.", family_uid=family_uid)
            uid = str(row["action_program_uid"])
            if not uid or uid in seen:
                raise ControlledWorldContractError("duplicate_action_program", "Action program UIDs must be unique per family.", family_uid=family_uid)
            seen.add(uid)
    support_cases = dict(value["development_support_cases"])
    minimums = dict(support_cases.get("per_family_minimums", {}))
    if minimums != {"supported": 1, "out_of_distribution": 1, "unsupported": 1}:
        raise ControlledWorldContractError("invalid_support_case_policy", "Development support/OOD/unsupported minimums are not exact.")
    if support_cases.get("retained_case_execution_status") != "retained_not_executed":
        raise ControlledWorldContractError("invalid_support_case_policy", "Support probes must be retained without execution.")
    if dict(value["historical_benchmark_provenance"]).get("sealed_content_opened") is not False:
        raise ControlledWorldContractError("sealed_access_violation", "Historical sealed content must remain unopened.")
    return value


def _source_config_identity(repository_root: Path) -> str:
    return sha256_file(repository_root / "Libs/configs/data/controlled_neural_world.yaml")


def _open_config_identity(repository_root: Path) -> str:
    return sha256_file(repository_root / "Libs/configs/data/controlled_world_open_action_identification.yaml")


def _role_seed(open_cfg: Mapping[str, object], role: str, family_index: int, latent_index: int) -> int:
    base = int(dict(dict(open_cfg["role_seed_namespaces"])[role])["base_seed"])
    return base + family_index * 1000 + latent_index


def _materialized_source(source: Mapping[str, object], family_uid: str, *, world_uid: str, lineage_uid: str, seed: int) -> dict[str, object]:
    value = deepcopy(dict(source))
    for family in value["families"]:
        if family["family_uid"] != family_uid:
            continue
        family["generator"]["seed"] = seed
        family["generator"]["subseeds"] = {"parameter_jitter": seed * 100 + 1, "observation_noise": seed * 100 + 2}
        family["generator"]["latent_generator_lineage_uid"] = lineage_uid
        family["world"]["world_uid"] = world_uid
        return value
    raise ControlledWorldContractError("unsupported_world_family", "Unknown family UID.", family_uid=family_uid)


def _action_for_program(family: Mapping[str, object], program: Mapping[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    action = deepcopy(dict(family["world_action"]["canonical"]))
    extension = deepcopy(dict(family["world_action"]["extension"]))
    intensity = deepcopy(dict(program["dose_or_intensity"]))
    duration = deepcopy(dict(program["duration"]))
    action["duration"] = duration
    extension["duration"] = duration
    extension["dose_or_intensity"] = intensity
    if action["intervention_type"] == "pharmacological":
        action["dose"] = float(intensity["value"])
        action["dose_unit"] = intensity["unit"]
    else:
        action["dose"] = None
        action["dose_unit"] = None
    return action, extension


def _latent_identity_basis(
    family: Mapping[str, object],
    *,
    role: str,
    latent_world_uid: str,
    lineage_uid: str,
    seed: int,
) -> dict[str, object]:
    generator = dict(family["generator"])
    return {
        "family_uid": family["family_uid"],
        "role": role,
        "latent_world_uid": latent_world_uid,
        "latent_generator_lineage_uid": lineage_uid,
        "seed": seed,
        "seed_policy": generator["seed_policy"],
        "subseeds": {"parameter_jitter": seed * 100 + 1, "observation_noise": seed * 100 + 2},
        "biological_context": family["world"]["biological_context"],
        "observation_policy": family["world"]["observation_policy"],
    }


def _supported_execution_row(
    *,
    source: Mapping[str, object],
    open_cfg: Mapping[str, object],
    family_uid: str,
    family_index: int,
    role: str,
    latent_index: int,
    program: Mapping[str, object],
    counterfactual_pair_uid: str,
    repository_root: Path,
) -> dict[str, object]:
    base_family = select_family_config(source, family_uid)
    seed = _role_seed(open_cfg, role, family_index, latent_index)
    latent_world_uid = f"{family_uid}_{role}_latent_world_{latent_index:04d}"
    lineage_uid = f"{family_uid}_{role}_action_identification_latent_lineage_{latent_index:04d}"
    program_uid = str(program["action_program_uid"])
    world_uid = f"{latent_world_uid}_{program_uid}_execution"
    materialized = _materialized_source(source, family_uid, world_uid=world_uid, lineage_uid=lineage_uid, seed=seed)
    materialized_family = select_family_config(materialized, family_uid)
    action, extension = _action_for_program(base_family, program)
    binding = bind_world_action(base_family, action, extension).to_dict()
    support = classify_world_action(base_family, binding["canonical_world_action"], binding["controlled_extension"]).to_dict()
    if support["status"] != "supported":
        raise ControlledWorldContractError("action_program_not_supported", "Declared supported action program is outside support.", family_uid=family_uid, action_program_uid=program_uid)
    config_file_identity = _source_config_identity(repository_root)
    instance, information_bindings = generate_runtime_instance(
        materialized,
        family_uid=family_uid,
        repository_root=repository_root,
        config_file_identity=config_file_identity,
        canonical_action=binding["canonical_world_action"],
        action_extension=binding["controlled_extension"],
    )
    provenance = dict(instance["generator_provenance"])
    identity = dict(instance["world_identity"])
    if identity["world_uid"] != world_uid or identity["latent_generator_lineage_uid"] != lineage_uid:
        raise ControlledWorldContractError("runtime_identity_mismatch", "Runtime identity did not preserve the requested latent/action execution identity.")
    observable_identity = canonical_sha256(instance)
    sequence_identity = action_sequence_identity(binding)
    replay_identity = stable_uid(
        "controlled_world_open_action_replay",
        {
            "world_uid": world_uid,
            "counterfactual_pair_uid": counterfactual_pair_uid,
            "action_sequence_identity": sequence_identity,
            "observable_output_identity": observable_identity,
        },
    )
    latent_basis = _latent_identity_basis(
        base_family,
        role=role,
        latent_world_uid=latent_world_uid,
        lineage_uid=lineage_uid,
        seed=seed,
    )
    return {
        "world_uid": world_uid,
        "latent_world_uid": latent_world_uid,
        "family_uid": family_uid,
        "role": role,
        "latent_generator_lineage_uid": lineage_uid,
        "seed": provenance["seed"],
        "seed_identity": canonical_sha256(latent_basis),
        "counterfactual_pair_uid": counterfactual_pair_uid,
        "action_program_uid": program_uid,
        "action_binding": binding,
        "action_binding_identity": binding["binding_uid"],
        "action_sequence_identity": sequence_identity,
        "action_definition_identity": canonical_sha256(program),
        "generator_implementation_uid": provenance["implementation_uid"],
        "generator_implementation_version": provenance["implementation_version"],
        "generator_code_identity": provenance["code_identity"],
        "config_content_identity": provenance["config_content_identity"],
        "config_file_identity": provenance["config_file_identity"],
        "shared_latent_config_identity": canonical_sha256(latent_basis),
        "biological_context_identity": canonical_sha256(identity["biological_context"]),
        "observation_policy_identity": canonical_sha256(materialized_family["world"]["observation_policy"]),
        "generator_subseeds": dict(materialized_family["generator"]["subseeds"]),
        "replay_identity": replay_identity,
        "observable_output_identity": observable_identity,
        "protocol_state_identity": stable_uid(
            "controlled_world_open_protocol_state",
            {"role": role, "latent_world_uid": latent_world_uid, "world_uid": world_uid, "action_program_uid": program_uid},
        ),
        "information_sequence_identity": information_sequence_identity(information_bindings),
        "support_status": support,
        "execution_status": "executed_supported",
        "failure_status": None,
        "access_counters": dict(ZERO_ACCESS_COUNTERS),
    }


def _support_case(
    source: Mapping[str, object],
    family_uid: str,
    kind: str,
    *,
    repository_root: Path,
) -> dict[str, object]:
    family = select_family_config(source, family_uid)
    action = deepcopy(dict(family["world_action"]["canonical"]))
    extension = deepcopy(dict(family["world_action"]["extension"]))
    attempted: dict[str, object]
    binding: dict[str, object] | None = None
    binding_identity: str | None = None
    failure: dict[str, object] | None = None
    if kind == "supported":
        attempted = {"canonical_world_action": action, "controlled_extension": extension}
        binding = bind_world_action(family, action, extension).to_dict()
        support = classify_world_action(family, binding["canonical_world_action"], binding["controlled_extension"]).to_dict()
        binding_identity = binding["binding_uid"]
    elif kind == "out_of_distribution":
        bounds = dict(family["support_domain"]["supported_intensity"])
        extension["dose_or_intensity"] = {"value": float(bounds["maximum"]) * 1.25, "unit": bounds["unit"]}
        if action["intervention_type"] == "pharmacological":
            action["dose"] = extension["dose_or_intensity"]["value"]
            action["dose_unit"] = extension["dose_or_intensity"]["unit"]
        attempted = {"canonical_world_action": action, "controlled_extension": extension}
        binding = bind_world_action(family, action, extension).to_dict()
        support = classify_world_action(family, binding["canonical_world_action"], binding["controlled_extension"]).to_dict()
        binding_identity = binding["binding_uid"]
    elif kind == "unsupported":
        route = "unsupported_intravenous_route" if action["route"] != "unsupported_intravenous_route" else "unsupported_extravascular_route"
        action["route"] = route
        extension["action_semantics"] = dict(extension["action_semantics"], route=route)
        attempted = {"canonical_world_action": action, "controlled_extension": extension}
        support = classify_world_action(family, action, extension).to_dict()
        try:
            binding = bind_world_action(family, action, extension).to_dict()
            binding_identity = binding["binding_uid"]
        except ControlledWorldContractError as exc:
            failure = exc.to_blocker()
    else:
        raise ControlledWorldContractError("invalid_support_case_policy", "Unknown support-case kind.", support_case_kind=kind)
    expected_status = "out_of_distribution" if kind == "out_of_distribution" else kind
    if support["status"] != expected_status:
        raise ControlledWorldContractError(
            "support_case_status_mismatch",
            "Development support case does not classify to its declared status.",
            family_uid=family_uid,
            support_case_kind=kind,
            observed_status=support["status"],
        )
    return {
        "support_case_uid": stable_uid(
            "controlled_world_open_support_case",
            {"family_uid": family_uid, "support_case_kind": kind, "attempted_action": attempted},
        ),
        "family_uid": family_uid,
        "support_case_kind": kind,
        "support_status": support,
        "execution_status": "retained_not_executed",
        "attempted_action": attempted,
        "action_binding": binding,
        "action_binding_identity": binding_identity,
        "reason_codes": list(support["reason_codes"]),
        "provenance": {
            "support_domain_ref": family["support_domain"]["support_domain_uid"],
            "repository_code_identity": sha256_file(repository_root / "Libs/utils/data/controlled_world_contracts.py"),
            "source": "controlled_world_open_action_identification_config",
        },
        "failure_record": failure,
        "access_counters": dict(ZERO_ACCESS_COUNTERS),
    }


def build_open_manifests(config: Mapping[str, object], *, repository_root: Path) -> dict[str, dict[str, object]]:
    source = validate_controlled_world_config(config)
    open_cfg = validate_open_package_config(config)
    manifests: dict[str, dict[str, object]] = {}
    pairing_authority = {
        "authority_uid": stable_uid("controlled_world_open_pairing_authority", dict(open_cfg["pairing_policy"])),
        **dict(open_cfg["pairing_policy"]),
    }
    for role in OPEN_ROLES:
        worlds: list[dict[str, object]] = []
        pairs: list[dict[str, object]] = []
        for family_index, family_uid in enumerate(FAMILY_UIDS):
            latent_count = int(dict(open_cfg["latent_worlds_per_family"])[role])
            programs = [dict(program) for program in dict(open_cfg["action_programs"])[family_uid]]
            for latent_index in range(latent_count):
                seed = _role_seed(open_cfg, role, family_index, latent_index)
                latent_world_uid = f"{family_uid}_{role}_latent_world_{latent_index:04d}"
                lineage_uid = f"{family_uid}_{role}_action_identification_latent_lineage_{latent_index:04d}"
                seed_identity = canonical_sha256(
                    _latent_identity_basis(
                        select_family_config(source, family_uid),
                        role=role,
                        latent_world_uid=latent_world_uid,
                        lineage_uid=lineage_uid,
                        seed=seed,
                    )
                )
                pair_uid = stable_uid(
                    "controlled_world_open_counterfactual_pair",
                    {
                        "family_uid": family_uid,
                        "role": role,
                        "latent_world_uid": latent_world_uid,
                        "latent_generator_lineage_uid": lineage_uid,
                        "seed_identity": seed_identity,
                        "action_program_uids": [program["action_program_uid"] for program in programs],
                    },
                )
                member_rows = [
                    _supported_execution_row(
                        source=source,
                        open_cfg=open_cfg,
                        family_uid=family_uid,
                        family_index=family_index,
                        role=role,
                        latent_index=latent_index,
                        program=program,
                        counterfactual_pair_uid=pair_uid,
                        repository_root=repository_root,
                    )
                    for program in programs
                ]
                worlds.extend(member_rows)
                pairs.append(
                    {
                        "counterfactual_pair_uid": pair_uid,
                        "role": role,
                        "family_uid": family_uid,
                        "latent_world_uid": latent_world_uid,
                        "latent_generator_lineage_uid": lineage_uid,
                        "seed_identity": seed_identity,
                        "member_world_uids": [row["world_uid"] for row in member_rows],
                        "action_program_uids": [row["action_program_uid"] for row in member_rows],
                        "approved_pair": True,
                        "shared_latent_config_identity": member_rows[0]["shared_latent_config_identity"],
                        "biological_context_identity": member_rows[0]["biological_context_identity"],
                        "observation_policy_identity": member_rows[0]["observation_policy_identity"],
                    }
                )
        support_cases = (
            [
                _support_case(source, family_uid, kind, repository_root=repository_root)
                for family_uid in FAMILY_UIDS
                for kind in ("supported", "out_of_distribution", "unsupported")
            ]
            if role == "development"
            else []
        )
        family_counts = {
            family_uid: {
                "latent_worlds": int(dict(open_cfg["latent_worlds_per_family"])[role]),
                "supported_executions": sum(1 for row in worlds if row["family_uid"] == family_uid),
                "counterfactual_pairs": sum(1 for row in pairs if row["family_uid"] == family_uid),
                "development_support_cases": sum(1 for row in support_cases if row["family_uid"] == family_uid),
            }
            for family_uid in FAMILY_UIDS
        }
        manifest_basis = {
            "role": role,
            "pairing_authority": pairing_authority,
            "counterfactual_pairs": pairs,
            "worlds": worlds,
            "development_support_cases": support_cases,
        }
        manifest = {
            "schema_version": OPEN_MANIFEST_SCHEMA_VERSION,
            "manifest_uid": stable_uid("controlled_world_open_action_manifest", manifest_basis),
            "role": role,
            "evidence_class": "controlled_world_non_empirical",
            "independent_unit": "latent_world",
            "family_uids": list(FAMILY_UIDS),
            "latent_world_count": len(pairs),
            "supported_execution_count": len(worlds),
            "support_case_count": len(support_cases),
            "family_counts": family_counts,
            "pairing_authority": pairing_authority,
            "counterfactual_pairs": pairs,
            "worlds": worlds,
            "development_support_cases": support_cases,
            "access_counters": dict(ZERO_ACCESS_COUNTERS),
            "closed_downstream_flags": dict(CLOSED_DOWNSTREAM_FLAGS),
            "content_identity": canonical_sha256(manifest_basis),
        }
        manifests[role] = manifest
    validate_open_manifests(manifests, config=config, repository_root=repository_root)
    return manifests


def _row_instance(row: Mapping[str, object], source: Mapping[str, object], repository_root: Path) -> dict[str, object]:
    seed = int(row["seed"])
    expected_subseeds = {"parameter_jitter": seed * 100 + 1, "observation_noise": seed * 100 + 2}
    if dict(row["generator_subseeds"]) != expected_subseeds:
        raise ControlledWorldContractError("seed_provenance_mismatch", "Recorded generator sub-seeds do not match the role seed identity.")
    materialized = _materialized_source(
        source,
        str(row["family_uid"]),
        world_uid=str(row["world_uid"]),
        lineage_uid=str(row["latent_generator_lineage_uid"]),
        seed=seed,
    )
    instance, _ = generate_runtime_instance(
        materialized,
        family_uid=str(row["family_uid"]),
        repository_root=repository_root,
        config_file_identity=str(row["config_file_identity"]),
        canonical_action=dict(dict(row["action_binding"])["canonical_world_action"]),
        action_extension=dict(dict(row["action_binding"])["controlled_extension"]),
    )
    return instance


def validate_open_manifests(
    manifests: Mapping[str, Mapping[str, object]],
    *,
    config: Mapping[str, object],
    repository_root: Path,
    replay: bool = True,
) -> list[dict[str, object]]:
    source = validate_controlled_world_config(config)
    open_cfg = validate_open_package_config(config)
    blockers: list[dict[str, object]] = []
    if set(manifests) != set(OPEN_ROLES):
        return [{"code": "missing_open_role", "roles": sorted(manifests)}]
    cross_role_values: dict[str, dict[str, set[str]]] = {
        field: {role: set() for role in OPEN_ROLES}
        for field in ("latent_world_uid", "latent_generator_lineage_uid", "seed_identity")
    }
    for role in OPEN_ROLES:
        manifest = dict(manifests[role])
        if manifest.get("schema_version") != OPEN_MANIFEST_SCHEMA_VERSION or manifest.get("role") != role:
            blockers.append({"code": "manifest_schema_or_role_mismatch", "role": role})
            continue
        if manifest.get("access_counters") != ZERO_ACCESS_COUNTERS or manifest.get("closed_downstream_flags") != CLOSED_DOWNSTREAM_FLAGS:
            blockers.append({"code": "open_gate_or_access_violation", "role": role})
        rows = [dict(row) for row in manifest.get("worlds", [])]
        pairs = [dict(row) for row in manifest.get("counterfactual_pairs", [])]
        support_cases = [dict(row) for row in manifest.get("development_support_cases", [])]
        if role != "development" and support_cases:
            blockers.append({"code": "support_cases_outside_development", "role": role})
        if role == "development":
            by_family_kind = {(case.get("family_uid"), case.get("support_case_kind")) for case in support_cases}
            for family_uid in FAMILY_UIDS:
                for kind in ("supported", "out_of_distribution", "unsupported"):
                    if (family_uid, kind) not in by_family_kind:
                        blockers.append({"code": "development_support_case_missing", "family_uid": family_uid, "support_case_kind": kind})
            for case in support_cases:
                if case.get("execution_status") != "retained_not_executed" or case.get("access_counters") != ZERO_ACCESS_COUNTERS:
                    blockers.append({"code": "support_case_execution_or_access_violation", "support_case_uid": case.get("support_case_uid")})
                if case.get("support_case_kind") in {"out_of_distribution", "unsupported"}:
                    if case.get("observable_output_identity") is not None or case.get("replay_identity") is not None:
                        blockers.append({"code": "support_case_fabricated_outcome", "support_case_uid": case.get("support_case_uid")})
        pair_members: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            if row.get("role") != role:
                blockers.append({"code": "row_role_mismatch", "world_uid": row.get("world_uid"), "role": role})
            if row.get("execution_status") != "executed_supported" or dict(row.get("support_status", {})).get("status") != "supported":
                blockers.append({"code": "unsupported_supervision_row", "world_uid": row.get("world_uid")})
            if row.get("access_counters") != ZERO_ACCESS_COUNTERS:
                blockers.append({"code": "nonzero_access_counter", "world_uid": row.get("world_uid")})
            binding = dict(row.get("action_binding", {}))
            if row.get("action_binding_identity") != binding.get("binding_uid"):
                blockers.append({"code": "action_binding_identity_mismatch", "world_uid": row.get("world_uid")})
            if binding and row.get("action_sequence_identity") != action_sequence_identity(binding):
                blockers.append({"code": "action_sequence_identity_mismatch", "world_uid": row.get("world_uid")})
            pair_members[str(row.get("counterfactual_pair_uid"))].append(row)
            for field in cross_role_values:
                cross_role_values[field][role].add(str(row.get(field)))
            if replay:
                try:
                    instance = _row_instance(row, source, repository_root)
                    if canonical_sha256(instance) != row.get("observable_output_identity"):
                        blockers.append({"code": "observable_replay_mismatch", "world_uid": row.get("world_uid")})
                    if dict(instance["action_binding"]) != binding:
                        blockers.append({"code": "action_binding_replay_mismatch", "world_uid": row.get("world_uid")})
                except Exception as exc:  # noqa: BLE001 - validation must retain exact blocker.
                    blockers.append({"code": "deterministic_replay_failed", "world_uid": row.get("world_uid"), "message": str(exc)})
        pair_ids = {str(pair.get("counterfactual_pair_uid")) for pair in pairs}
        if pair_ids != set(pair_members):
            blockers.append({"code": "pair_authority_membership_mismatch", "role": role})
        seen_world_members: set[str] = set()
        for pair in pairs:
            pair_uid = str(pair["counterfactual_pair_uid"])
            members = pair_members.get(pair_uid, [])
            if len(members) != 2:
                blockers.append({"code": "counterfactual_pair_cardinality_invalid", "counterfactual_pair_uid": pair_uid})
                continue
            if set(pair["member_world_uids"]) != {str(row["world_uid"]) for row in members}:
                blockers.append({"code": "counterfactual_pair_member_uid_mismatch", "counterfactual_pair_uid": pair_uid})
            if seen_world_members & set(pair["member_world_uids"]):
                blockers.append({"code": "duplicate_pair_membership", "counterfactual_pair_uid": pair_uid})
            seen_world_members.update(str(uid) for uid in pair["member_world_uids"])
            for field in dict(open_cfg["pairing_policy"])["require_same_fields"]:
                if len({canonical_sha256(row.get(field)) for row in members}) != 1:
                    blockers.append({"code": "counterfactual_pair_shared_field_mismatch", "counterfactual_pair_uid": pair_uid, "field": field})
            for field in dict(open_cfg["pairing_policy"])["require_distinct_fields"]:
                if len({canonical_sha256(row.get(field)) for row in members}) != len(members):
                    blockers.append({"code": "counterfactual_pair_distinct_field_mismatch", "counterfactual_pair_uid": pair_uid, "field": field})
        basis = {
            "role": role,
            "pairing_authority": manifest.get("pairing_authority"),
            "counterfactual_pairs": pairs,
            "worlds": rows,
            "development_support_cases": support_cases,
        }
        if manifest.get("content_identity") != canonical_sha256(basis):
            blockers.append({"code": "manifest_content_identity_mismatch", "role": role})
        expected_latent = sum(int(dict(open_cfg["latent_worlds_per_family"])[role]) for _ in FAMILY_UIDS)
        if len(pairs) != expected_latent or len(rows) != expected_latent * 2:
            blockers.append({"code": "role_cardinality_mismatch", "role": role, "expected_latent_worlds": expected_latent})
    for left_index, left in enumerate(OPEN_ROLES):
        for right in OPEN_ROLES[left_index + 1:]:
            for field, by_role in cross_role_values.items():
                overlap = by_role[left] & by_role[right]
                if overlap:
                    blockers.append({"code": "cross_role_latent_overlap", "field": field, "left_role": left, "right_role": right, "overlap_count": len(overlap)})
    return blockers


def build_package_receipt(
    manifests: Mapping[str, Mapping[str, object]],
    config: Mapping[str, object],
) -> dict[str, object]:
    open_cfg = validate_open_package_config(config)
    paths = dict(open_cfg["output_paths"])
    manifest_paths = {role: paths[f"{role}_manifest"] for role in OPEN_ROLES}
    manifest_identities = {role: str(manifests[role]["content_identity"]) for role in OPEN_ROLES}
    basis = {
        "manifests": manifest_paths,
        "manifest_content_identities": manifest_identities,
        "historical_benchmark_provenance": open_cfg["historical_benchmark_provenance"],
        "closed_downstream_flags": CLOSED_DOWNSTREAM_FLAGS,
    }
    return {
        "schema_version": OPEN_PACKAGE_RECEIPT_SCHEMA_VERSION,
        "receipt_uid": stable_uid("controlled_world_open_package_receipt", basis),
        "evidence_class": "controlled_world_non_empirical",
        "terminal_decision": OPEN_PACKAGE_POSITIVE_TERMINAL,
        "open_data_qualification": "pending",
        "training_authorized": False,
        "sufficient_training": False,
        "twin_qualification_pass": False,
        "formal_qualification": False,
        "agent_work_open": False,
        "test_manifest_open": False,
        "manifests": manifest_paths,
        "manifest_content_identities": manifest_identities,
        "historical_benchmark_provenance": dict(open_cfg["historical_benchmark_provenance"]),
        "access_counters": dict(ZERO_ACCESS_COUNTERS),
        "closed_downstream_flags": dict(CLOSED_DOWNSTREAM_FLAGS),
        "deterministic_replay": {
            "required": True,
            "supported_execution_rows": sum(len(manifests[role]["worlds"]) for role in OPEN_ROLES),
            "status": "validated_during_construction",
        },
        "construction_audit_path": paths["construction_audit"],
        "construction_report_path": paths["construction_report"],
    }


def build_controlled_world_open_package(
    *,
    config: Mapping[str, object],
    repository_root: Path,
    artifact_root: Path,
) -> dict[str, Path]:
    manifests = build_open_manifests(config, repository_root=repository_root)
    receipt = build_package_receipt(manifests, config)
    paths = dict(validate_open_package_config(config)["output_paths"])
    written: dict[str, Path] = {}
    for role in OPEN_ROLES:
        path = resolve_declared_output(artifact_root, paths[f"{role}_manifest"], label=f"{role}_manifest")
        write_json(path, manifests[role])
        written[f"{role}_manifest"] = path
    receipt_path = resolve_declared_output(artifact_root, paths["package_receipt"], label="package_receipt")
    write_json(receipt_path, receipt)
    written["package_receipt"] = receipt_path
    return written


def build_construction_audit(
    manifests: Mapping[str, Mapping[str, object]],
    receipt: Mapping[str, object],
    blockers: Sequence[Mapping[str, object]],
    *,
    config: Mapping[str, object],
) -> dict[str, object]:
    open_cfg = validate_open_package_config(config)
    status = OPEN_PACKAGE_POSITIVE_TERMINAL if not blockers else OPEN_PACKAGE_NEGATIVE_TERMINAL
    role_summary = {
        role: {
            "latent_world_count": manifests.get(role, {}).get("latent_world_count", 0),
            "supported_execution_count": manifests.get(role, {}).get("supported_execution_count", 0),
            "support_case_count": manifests.get(role, {}).get("support_case_count", 0),
            "family_counts": manifests.get(role, {}).get("family_counts", {}),
        }
        for role in OPEN_ROLES
    }
    return {
        "schema_version": "neuronsifter-controlled-world-open-package-construction-audit-v1",
        "terminal_decision": status,
        "open_data_qualification": "pending",
        "training_authorized": False,
        "sufficient_training": False,
        "twin_qualification_pass": False,
        "formal_qualification": False,
        "agent_work_open": False,
        "test_manifest_open": False,
        "blocker_count": len(blockers),
        "blockers": [dict(row) for row in blockers],
        "role_summary": role_summary,
        "package_receipt_uid": receipt.get("receipt_uid"),
        "historical_benchmark_provenance": dict(open_cfg["historical_benchmark_provenance"]),
        "access_counters": dict(ZERO_ACCESS_COUNTERS),
        "closed_downstream_flags": dict(CLOSED_DOWNSTREAM_FLAGS),
        "forbidden_material_access": {
            "complete_test_manifest_content_opened": 0,
            "sealed_partition_content_opened": 0,
            "unopened_access_receipt_content_opened": 0,
            "protected_holdout_access": 0,
            "six_context_outcome_access": 0,
        },
    }


def construction_report(audit: Mapping[str, object]) -> str:
    lines = [
        "# Controlled World Open Action Package Construction",
        "",
        f"Terminal: {audit['terminal_decision']}",
        "",
        "Evidence class: controlled-world evidence only; not empirical, therapeutic, or clinical evidence.",
        "Open-data qualification: pending.",
        "Training authorized: false.",
        "Sufficient training: false.",
        "Twin qualification pass: false.",
        "Formal qualification: false.",
        "Agent, Tool, MCP, AgentTrace, Memory, and Co-Evolution gates: closed.",
        "Access counters: six_context_outcome_access=0, protected_holdout_access=0, test_outcome_access=0.",
        "",
        "## Role Summary",
    ]
    for role, summary in dict(audit["role_summary"]).items():
        lines.append(
            f"- {role}: latent_world_count={summary['latent_world_count']}, "
            f"supported_execution_count={summary['supported_execution_count']}, "
            f"support_case_count={summary['support_case_count']}"
        )
    if audit["blockers"]:
        lines.extend(["", "## Blockers"])
        for blocker in audit["blockers"]:
            lines.append(f"- {blocker.get('code')}: {blocker}")
    return "\n".join(lines) + "\n"


def write_construction_audit_and_report(
    *,
    config: Mapping[str, object],
    artifact_root: Path,
    manifests: Mapping[str, Mapping[str, object]],
    receipt: Mapping[str, object],
    blockers: Sequence[Mapping[str, object]],
) -> dict[str, Path]:
    paths = dict(validate_open_package_config(config)["output_paths"])
    audit = build_construction_audit(manifests, receipt, blockers, config=config)
    audit_path = resolve_declared_output(artifact_root, paths["construction_audit"], label="construction_audit")
    report_path = resolve_declared_output(artifact_root, paths["construction_report"], label="construction_report")
    write_json(audit_path, audit)
    write_text_atomic(report_path, construction_report(audit))
    return {"construction_audit": audit_path, "construction_report": report_path}
