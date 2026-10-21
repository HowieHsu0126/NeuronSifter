"""Controlled neural-world benchmark split and OOD manifest construction."""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from Libs.utils.data.controlled_world_contracts import (
    FAMILY_UIDS,
    ControlledWorldContractError,
    bind_world_action,
    classify_world_action,
    select_family_config,
    validate_controlled_world_config,
)
from Libs.utils.data.controlled_world_runtime import generate_runtime_instance, sha256_file
from Libs.utils.data.trajectory_contracts import canonical_sha256, stable_uid

BENCHMARK_CONFIG_SCHEMA_VERSION = "neuronsifter-controlled-world-benchmark-config-v1"
BENCHMARK_MANIFEST_SCHEMA_VERSION = "neuronsifter-controlled-world-benchmark-manifest-v1"
OOD_MANIFEST_SCHEMA_VERSION = "neuronsifter-controlled-world-ood-manifest-v1"
PARTITIONS = ("twin_qualification", "agent_decision", "component_mechanism", "coevolution_robustness")
OOD_CLASSES = ("parametric_ood", "intervention_ood", "combination_ood", "observation_shift_ood", "structural_ood")
ROLES = ("training", "development", "calibration", "test")
TERMINALS = ("CONTROLLED_NEURAL_WORLD_BENCHMARK_QUALIFIED", "CONTROLLED_NEURAL_WORLD_BENCHMARK_NOT_QUALIFIED")

FORBIDDEN_SOURCE_MARKERS = (
    "operational_twin",
    "downstream_agent",
    "agent_trace",
    "tool_memory",
    "coevolution_execution",
    "therapeutic_evidence",
    "clinical_endpoint",
    "supervised_development_data",
    "checkpoint_rescue",
    "simulator_asset_splicing",
    "mechanism_search",
    "prior_intake_reopen",
    "six_context_outcome",
    "protected_holdout",
)


def benchmark_config(config: Mapping[str, object]) -> dict[str, Any]:
    value = config.get("controlled_world_benchmark_qualification") if "controlled_world_benchmark_qualification" in config else config
    if not isinstance(value, Mapping):
        raise ControlledWorldContractError("missing_benchmark_config", "Controlled-world benchmark qualification config is missing.")
    return deepcopy(dict(value))


def validate_benchmark_config(config: Mapping[str, object]) -> dict[str, Any]:
    value = benchmark_config(config)
    required = {
        "schema_version",
        "evidence_class",
        "authority_role",
        "real_world_formal_agent_evaluation",
        "allowed_family_uids",
        "access_policy",
        "source_firewall",
        "split_policy",
        "latent_lineage_policy",
        "action_grid",
        "ood_policy",
        "headroom_policy",
        "statistical_policy",
        "sample_size_policy",
        "freeze_policy",
        "terminal_policy",
        "output_paths",
    }
    if set(value) != required:
        raise ControlledWorldContractError("invalid_benchmark_config", "Benchmark config keys are not the frozen contract.", fields=sorted(set(value) ^ required))
    if value["schema_version"] != BENCHMARK_CONFIG_SCHEMA_VERSION or value["evidence_class"] != "controlled_world_non_empirical":
        raise ControlledWorldContractError("invalid_benchmark_config", "Benchmark config schema or evidence class is invalid.")
    if value["authority_role"] != "benchmark_harness" or value["real_world_formal_agent_evaluation"] is not False:
        raise ControlledWorldContractError("benchmark_authority_invalid", "Benchmark authority must remain harness-only and non-real-world.")
    if list(value["allowed_family_uids"]) != list(FAMILY_UIDS):
        raise ControlledWorldContractError("unsupported_world_family", "Benchmark must use the exact Round 29 three-family authority.")
    access = dict(value["access_policy"])
    if access != {"six_context_outcome_access": 0, "protected_holdout_access": 0, "test_outcome_access": 0, "protected_result_access": 0}:
        raise ControlledWorldContractError("protected_access_nonzero", "Benchmark access counters must remain exactly zero.")
    firewall = dict(value["source_firewall"])
    if firewall.get("strict_public_only_flag") is not True or firewall.get("controlled_access_required") is not False:
        raise ControlledWorldContractError("public_boundary_violation", "Controlled benchmark cannot require controlled access.")
    if tuple(firewall.get("prohibited_source_routes", ())) != FORBIDDEN_SOURCE_MARKERS or firewall.get("fail_on_prohibited_reference") is not True:
        raise ControlledWorldContractError("prohibited_source_policy_invalid", "Prohibited source routes are not the frozen fail-closed list.")
    split = dict(value["split_policy"])
    if tuple(split.get("roles", ())) != ROLES or tuple(split.get("test_partitions", ())) != PARTITIONS:
        raise ControlledWorldContractError("invalid_split_policy", "Split roles or sealed test partitions are not canonical.")
    if int(split["minimum_test_worlds_per_family_per_partition"]) < 1 or int(split["maximum_test_worlds_per_family_per_partition"]) < int(split["minimum_test_worlds_per_family_per_partition"]):
        raise ControlledWorldContractError("invalid_split_policy", "Test partition world counts are infeasible.")
    if tuple(value["ood_policy"].get("classes", ())) != OOD_CLASSES:
        raise ControlledWorldContractError("invalid_ood_policy", "All five benchmark OOD classes are required in canonical order.")
    stats = dict(value["statistical_policy"])
    if stats.get("unit") != "independent_world" or stats.get("bootstrap_method") != "family_stratified_paired_bootstrap":
        raise ControlledWorldContractError("invalid_statistical_policy", "Independent world and family-stratified paired bootstrap are required.")
    handling = dict(stats.get("failure_and_undefined_handling", {}))
    if not all(handling.get(key) is True for key in ("retain_every_case", "failed_world_is_not_dropped", "undefined_world_is_not_imputed")):
        raise ControlledWorldContractError("invalid_failure_retention", "Failure and undefined cases must be retained.")
    terminal = dict(value["terminal_policy"])
    if tuple(terminal.get("allowed_terminals", ())) != TERMINALS:
        raise ControlledWorldContractError("invalid_terminal_policy", "Benchmark terminal vocabulary is not exact.")
    _check_semantic_paths(value)
    return value


def _check_semantic_paths(value: Mapping[str, object]) -> None:
    banned = ("m1", "m2", "m3", "phase", "stage")
    for name, raw_path in dict(value["output_paths"]).items():
        path = Path(str(raw_path))
        if path.is_absolute() or ".." in path.parts:
            raise ControlledWorldContractError("invalid_output_path", "Benchmark output paths must be repository-relative.", field=name)
        lowered = path.as_posix().lower()
        if any(token in lowered for token in banned):
            raise ControlledWorldContractError("nonsemantic_artifact_name", "Benchmark artifact names must be semantic.", field=name)


def _role_count(config: Mapping[str, object], role: str) -> int:
    split = dict(config["split_policy"])
    if role == "test":
        return int(split["minimum_test_worlds_per_family_per_partition"])
    return int(dict(split["worlds_per_family"])[role])


def _seed_base(config: Mapping[str, object], role_or_partition: str) -> int:
    return int(dict(config["split_policy"])["seed_bases"][role_or_partition])


def _materialized_config(source: Mapping[str, object], family_uid: str, role: str, index: int, seed: int) -> dict[str, object]:
    value = deepcopy(dict(source))
    for family in value["families"]:
        if family["family_uid"] != family_uid:
            continue
        family["generator"]["seed"] = seed
        family["generator"]["subseeds"] = {"parameter_jitter": seed * 100 + 1, "observation_noise": seed * 100 + 2}
        family["generator"]["latent_generator_lineage_uid"] = f"{family_uid}_{role}_latent_lineage_{index:04d}"
        family["world"]["world_uid"] = f"{family_uid}_{role}_world_{index:04d}"
        return value
    raise ControlledWorldContractError("unsupported_world_family", "Unknown family UID.", family_uid=family_uid)


def _world_record(source: Mapping[str, object], bench: Mapping[str, object], family_uid: str, role: str, index: int, repository_root: Path, *, partition: str | None = None) -> dict[str, object]:
    seed_key = partition or role
    seed = _seed_base(bench, seed_key) + index
    materialized = _materialized_config(source, family_uid, partition or role, index, seed)
    config_file_identity = canonical_sha256(select_family_config(materialized, family_uid))
    instance, information_bindings = generate_runtime_instance(materialized, family_uid=family_uid, repository_root=repository_root, config_file_identity=config_file_identity)
    identity = dict(instance["world_identity"])
    provenance = dict(instance["generator_provenance"])
    binding = dict(instance["action_binding"])
    record = {
        "world_uid": identity["world_uid"],
        "family_uid": family_uid,
        "role": role,
        "partition": partition,
        "latent_generator_lineage_uid": identity["latent_generator_lineage_uid"],
        "generator_implementation_uid": provenance["implementation_uid"],
        "generator_implementation_version": provenance["implementation_version"],
        "generator_code_identity": provenance["code_identity"],
        "config_content_identity": provenance["config_content_identity"],
        "seed": provenance["seed"],
        "seed_identity": provenance["seed_identity"],
        "action_sequence_identity": canonical_sha256(binding),
        "information_sequence_identity": canonical_sha256(information_bindings),
        "replay_identity": stable_uid("controlled_world_benchmark_replay", {"world_uid": identity["world_uid"], "observable_output_identity": canonical_sha256(instance)}),
        "support_status": dict(instance["support_status"]),
        "access_counters": dict(instance["access_counters"], test_outcome_access=0),
        "observable_output_identity": canonical_sha256(instance),
        "mutable_protocol_state_uid": stable_uid("controlled_world_protocol_state", {"role": role, "partition": partition, "world_uid": identity["world_uid"]}),
    }
    return record


def build_split_manifests(config: Mapping[str, object], *, repository_root: Path) -> dict[str, dict[str, object]]:
    source = validate_controlled_world_config(config)
    bench = validate_benchmark_config(config)
    manifests: dict[str, dict[str, object]] = {}
    for role in ("training", "development", "calibration"):
        worlds = [_world_record(source, bench, family_uid, role, idx, repository_root) for family_uid in FAMILY_UIDS for idx in range(_role_count(bench, role))]
        manifests[role] = _manifest(role, worlds, bench)
    test_worlds = []
    for partition in PARTITIONS:
        for family_uid in FAMILY_UIDS:
            for idx in range(_role_count(bench, "test")):
                test_worlds.append(_world_record(source, bench, family_uid, "test", idx, repository_root, partition=partition))
    manifests["test"] = _manifest("test", test_worlds, bench, sealed_partitions=list(PARTITIONS))
    validate_lineage_separation(manifests)
    return manifests


def _manifest(role: str, worlds: Sequence[Mapping[str, object]], bench: Mapping[str, object], *, sealed_partitions: list[str] | None = None) -> dict[str, object]:
    counts = {family_uid: sum(1 for row in worlds if row["family_uid"] == family_uid) for family_uid in FAMILY_UIDS}
    basis = {"role": role, "worlds": list(worlds), "policy": bench["split_policy"]}
    return {
        "schema_version": BENCHMARK_MANIFEST_SCHEMA_VERSION,
        "manifest_uid": stable_uid("controlled_world_benchmark_manifest", basis),
        "role": role,
        "evidence_class": "controlled_world_non_empirical",
        "independent_unit": "world",
        "family_uids": list(FAMILY_UIDS),
        "world_count": len(worlds),
        "family_counts": counts,
        "sealed_partitions": sealed_partitions or [],
        "access_counters": {"six_context_outcome_access": 0, "protected_holdout_access": 0, "test_outcome_access": 0},
        "worlds": list(worlds),
        "content_identity": canonical_sha256({"role": role, "worlds": list(worlds)}),
    }


def validate_lineage_separation(manifests: Mapping[str, Mapping[str, object]]) -> None:
    seen_by_role: dict[str, set[object]] = {"world_uid": set(), "seed_identity": set(), "latent_generator_lineage_uid": set()}
    for role, manifest in manifests.items():
        for row in manifest["worlds"]:
            for field in seen_by_role:
                value = row[field]
                if value in seen_by_role[field]:
                    raise ControlledWorldContractError("lineage_overlap", "Split manifests are not lineage-separated.", field=field, role=role)
                seen_by_role[field].add(value)
    test = manifests.get("test")
    if test:
        by_partition: dict[str, dict[str, set[object]]] = {}
        for row in test["worlds"]:
            partition = str(row["partition"])
            by_partition.setdefault(partition, {"world_uid": set(), "seed_identity": set(), "latent_generator_lineage_uid": set(), "mutable_protocol_state_uid": set()})
            for field, values in by_partition[partition].items():
                values.add(row[field])
        for left in PARTITIONS:
            for right in PARTITIONS:
                if left >= right:
                    continue
                for field in ("world_uid", "seed_identity", "latent_generator_lineage_uid", "mutable_protocol_state_uid"):
                    if by_partition[left][field] & by_partition[right][field]:
                        raise ControlledWorldContractError("sealed_partition_overlap", "Sealed test partitions overlap.", field=field)


def build_ood_manifest(config: Mapping[str, object], split_manifests: Mapping[str, Mapping[str, object]], *, repository_root: Path) -> dict[str, object]:
    source = validate_controlled_world_config(config)
    bench = validate_benchmark_config(config)
    records: list[dict[str, object]] = []
    per_class = int(dict(bench["ood_policy"])["worlds_per_family_per_class"])
    for family_uid in FAMILY_UIDS:
        family = select_family_config(source, family_uid)
        for ood_class in OOD_CLASSES:
            for idx in range(per_class):
                records.append(_ood_record(family, ood_class, idx, repository_root))
    manifest = {
        "schema_version": OOD_MANIFEST_SCHEMA_VERSION,
        "manifest_uid": stable_uid("controlled_world_ood_manifest", {"records": records}),
        "evidence_class": "controlled_world_non_empirical",
        "ood_classes": list(OOD_CLASSES),
        "family_uids": list(FAMILY_UIDS),
        "record_count": len(records),
        "records": records,
        "access_counters": {"six_context_outcome_access": 0, "protected_holdout_access": 0, "test_outcome_access": 0},
        "content_identity": canonical_sha256(records),
    }
    # Reference split_manifests so caller cannot accidentally build OOD before split validation.
    validate_lineage_separation(split_manifests)
    return manifest


def _ood_record(family: Mapping[str, object], ood_class: str, idx: int, repository_root: Path) -> dict[str, object]:
    action = deepcopy(family["world_action"]["canonical"])
    extension = deepcopy(family["world_action"]["extension"])
    support = dict(family["support_domain"])
    if ood_class in {"parametric_ood", "intervention_ood"}:
        bounds = dict(support["supported_intensity"])
        extension["dose_or_intensity"]["value"] = float(bounds["maximum"]) * (1.20 + 0.01 * idx)
        if action["intervention_type"] == "pharmacological":
            action["dose"] = extension["dose_or_intensity"]["value"]
    elif ood_class == "combination_ood":
        extension["combination"] = ["synthetic_second_agent"]
    elif ood_class == "observation_shift_ood":
        bounds = dict(support["supported_duration"])
        extension["duration"]["value"] = float(bounds["maximum"]) * (1.10 + 0.01 * idx)
        action["duration"]["value"] = extension["duration"]["value"]
    elif ood_class == "structural_ood":
        extension["biological_context"] = dict(extension["biological_context"], region="noncanonical_region")
        action["state_scope"] = dict(action["state_scope"], region=["noncanonical_region"])
    else:
        raise ControlledWorldContractError("invalid_ood_policy", "Unknown OOD class.")
    try:
        binding = bind_world_action(family, action, extension)
        support_decision = classify_world_action(family, binding.canonical_world_action, binding.controlled_extension).to_dict()
        failure = None
    except ControlledWorldContractError as exc:
        support_decision = {"status": "unsupported", "reason_codes": [exc.code], "support_domain_ref": support["support_domain_uid"]}
        failure = exc.to_blocker()
    return {
        "ood_record_uid": stable_uid("controlled_world_ood_case", {"family_uid": family["family_uid"], "ood_class": ood_class, "idx": idx}),
        "family_uid": family["family_uid"],
        "ood_class": ood_class,
        "support_status": support_decision,
        "forced_execution_status": "retained_not_executed" if support_decision["status"] != "supported" else "supported_control",
        "failure_record": failure,
        "access_counters": {"six_context_outcome_access": 0, "protected_holdout_access": 0, "test_outcome_access": 0},
        "repository_code_identity": sha256_file(repository_root / "Libs/utils/data/controlled_world_runtime.py"),
    }
