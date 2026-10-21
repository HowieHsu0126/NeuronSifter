from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from Libs.scripts.builders.build_controlled_neural_world import build_controlled_neural_world
from Libs.scripts.validators.validate_controlled_neural_world import validate_controlled_neural_world
from Libs.utils.data.action_observation_contracts import validate_world_action
from Libs.utils.data.controlled_world_contracts import (
    CLAIM_BOUNDARY,
    FAMILY_UIDS,
    ControlledWorldContractError,
    bind_information_action,
    bind_world_action,
    ensure_no_runtime_secrets,
    select_family_config,
    validate_controlled_world_config,
    validate_family_manifest,
    validate_runtime_instance,
)
from Libs.utils.data.controlled_world_runtime import (
    evaluate_harness_true_utility,
    generate_all_family_packages,
    generate_runtime_and_manifest,
    generate_runtime_instance,
    load_harness_oracle,
    replay_runtime_instance,
    sha256_file,
    simulate_harness_trajectory,
)
from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.data.trajectory_contracts import canonical_sha256, stable_uid


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"
WORLD_SCHEMA_PATH = REPO_ROOT / "Libs/configs/schema/controlled_neural_world.schema.json"
REPLAY_SCHEMA_PATH = REPO_ROOT / "Libs/configs/schema/controlled_world_replay_manifest.schema.json"
CONTROLLED_CONFIG_PATH = REPO_ROOT / "Libs/configs/data/controlled_neural_world.yaml"
CONFIG_FILE_IDENTITY = sha256_file(CONTROLLED_CONFIG_PATH)
HASH = "a" * 64


@pytest.fixture
def config() -> dict[str, object]:
    return load_config(CONFIG_PATH)


def _source(config: dict[str, object]) -> dict[str, object]:
    return config["controlled_neural_world"]


def _family(config: dict[str, object], family_uid: str) -> dict[str, object]:
    return select_family_config(_source(config), family_uid)


def _instance(config: dict[str, object], family_uid: str) -> dict[str, object]:
    value, _ = generate_runtime_instance(
        config,
        family_uid=family_uid,
        repository_root=REPO_ROOT,
        config_file_identity=CONFIG_FILE_IDENTITY,
    )
    return value


def _packages(config: dict[str, object]) -> dict[str, tuple[dict[str, object], dict[str, object]]]:
    return generate_all_family_packages(
        config,
        repository_root=REPO_ROOT,
        config_file_identity=CONFIG_FILE_IDENTITY,
    )


def _error_code(error: pytest.ExceptionInfo[ControlledWorldContractError]) -> str:
    return error.value.code


def _set_nested(value: dict[str, object], path: tuple[object, ...], replacement: object) -> None:
    current: object = value
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = replacement


def _rehash_family_manifest(manifest: dict[str, object]) -> None:
    manifest["manifest_uid"] = stable_uid(
        "controlled_world_family_manifest",
        {
            "source_config_identity": manifest["source_config_identity"],
            "families": manifest["families"],
        },
    )


def _alternative_action(family: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    action = deepcopy(family["world_action"]["canonical"])
    extension = deepcopy(family["world_action"]["extension"])
    bounds = family["support_domain"]["supported_intensity"]
    alternative = float(bounds["minimum"]) + 0.2 * (float(bounds["maximum"]) - float(bounds["minimum"]))
    extension["dose_or_intensity"]["value"] = alternative
    if action["intervention_type"] == "pharmacological":
        action["dose"] = alternative
    return action, extension


def test_build_validate_and_replay_real_config(tmp_path: Path) -> None:
    paths = build_controlled_neural_world(
        config_path=CONFIG_PATH,
        repository_root=REPO_ROOT,
        artifact_root=tmp_path,
    )
    audit = validate_controlled_neural_world(
        config_path=CONFIG_PATH,
        repository_root=REPO_ROOT,
        artifact_root=tmp_path,
    )

    assert len(paths) == 7
    assert all(path.is_file() for path in paths.values())
    assert audit["status"] == "passed"
    assert audit["blocker_count"] == 0
    assert audit["family_count"] == 3
    assert all(row["status"] == "passed" for row in audit["family_results"])
    assert audit["access_counters"] == {"six_context_outcome_access": 0, "protected_holdout_access": 0}
    assert audit["claim_boundary"] == CLAIM_BOUNDARY


def test_config_and_generated_artifacts_validate_against_registered_schemas(
    config: dict[str, object], tmp_path: Path
) -> None:
    paths = build_controlled_neural_world(
        config_path=CONFIG_PATH,
        repository_root=REPO_ROOT,
        artifact_root=tmp_path,
    )
    world_schema = json.loads(WORLD_SCHEMA_PATH.read_text(encoding="utf-8"))
    replay_schema = json.loads(REPLAY_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(world_schema)
    Draft202012Validator.check_schema(replay_schema)
    Draft202012Validator(world_schema).validate(_source(config))
    for name, path in paths.items():
        value = json.loads(path.read_text(encoding="utf-8"))
        schema = replay_schema if name.endswith("replay_manifest") else world_schema
        Draft202012Validator(schema).validate(value)


def test_exact_three_family_contract_is_semantically_distinct(config: dict[str, object]) -> None:
    source = validate_controlled_world_config(config)
    assert [row["family_uid"] for row in source["families"]] == list(FAMILY_UIDS)
    observable_sets = {
        family["family_uid"]: tuple(row["variable"] for row in family["world"]["observable_variables"])
        for family in source["families"]
    }
    parameter_sets = {
        family["family_uid"]: frozenset(family["generator"]["parameters"])
        for family in source["families"]
    }
    assert len(set(observable_sets.values())) == 3
    assert len(set(parameter_sets.values())) == 3
    assert observable_sets["calcium_synaptic_world"] == (
        "intracellular_calcium",
        "release_probability",
        "releasable_vesicle_fraction",
        "postsynaptic_current",
    )
    assert observable_sets["molecular_pharmacological_causal_world"] == (
        "free_agent_exposure",
        "target_occupancy",
        "downstream_signal",
        "biologic_response",
    )


@pytest.mark.parametrize("family_uid", FAMILY_UIDS)
def test_deterministic_replay_is_content_identical(config: dict[str, object], family_uid: str) -> None:
    first_instance, first_manifest = generate_runtime_and_manifest(
        config,
        family_uid=family_uid,
        repository_root=REPO_ROOT,
        config_file_identity=CONFIG_FILE_IDENTITY,
    )
    second_instance, second_manifest = generate_runtime_and_manifest(
        config,
        family_uid=family_uid,
        repository_root=REPO_ROOT,
        config_file_identity=CONFIG_FILE_IDENTITY,
    )
    replayed = replay_runtime_instance(config, first_manifest, repository_root=REPO_ROOT)

    assert canonical_sha256(first_instance) == canonical_sha256(second_instance)
    assert first_manifest == second_manifest
    assert replayed == first_instance


def test_family_generation_is_order_independent(config: dict[str, object]) -> None:
    forward = generate_all_family_packages(
        config,
        repository_root=REPO_ROOT,
        config_file_identity=CONFIG_FILE_IDENTITY,
    )
    reverse = generate_all_family_packages(
        config,
        repository_root=REPO_ROOT,
        config_file_identity=CONFIG_FILE_IDENTITY,
        family_uids=tuple(reversed(FAMILY_UIDS)),
    )

    for family_uid in FAMILY_UIDS:
        assert forward[family_uid] == reverse[family_uid]


@pytest.mark.parametrize("family_uid", FAMILY_UIDS)
def test_runtime_is_incomplete_noisy_and_contains_no_harness_state(
    config: dict[str, object], family_uid: str
) -> None:
    family = _family(config, family_uid)
    instance = _instance(config, family_uid)
    oracle = load_harness_oracle(config, family_uid=family_uid, reader_role="benchmark_harness")
    binding = bind_world_action(family)
    hidden_trajectory = simulate_harness_trajectory(family, oracle, binding.to_dict())

    ensure_no_runtime_secrets(instance)
    assert instance["authority_role"] == "runtime_observable"
    assert any(row["missing_observables"] for row in instance["observations"])
    observed_differences = []
    for index, row in enumerate(instance["observations"]):
        for variable, measurement in row["measurements"].items():
            observed_differences.append(measurement["value"] - hidden_trajectory[index][variable])
    assert any(abs(value) > 0 for value in observed_differences)
    serialized = json.dumps(instance, sort_keys=True)
    for forbidden in ("oracle_state", "oracle_parameters", "hidden_optimum", "true_utility", "protected_outcome"):
        assert forbidden not in serialized


@pytest.mark.parametrize("family_uid", FAMILY_UIDS)
def test_canonical_world_action_and_extension_round_trip(config: dict[str, object], family_uid: str) -> None:
    family = _family(config, family_uid)
    binding = bind_world_action(family)
    action = binding.canonical_world_action
    semantics = binding.controlled_extension["action_semantics"]

    assert validate_world_action(action) == []
    assert binding.controlled_extension["modality"] == action["intervention_type"]
    assert binding.controlled_extension["duration"] == action["duration"]
    assert semantics["target_set"] == action["target_set"]
    assert semantics["agent_set"] == action["agent_set"]
    assert semantics["schedule"] == action["schedule"]
    assert semantics["route"] == action["route"]
    if family_uid == "molecular_pharmacological_causal_world":
        assert action["dose"] == binding.controlled_extension["dose_or_intensity"]["value"]
        assert action["dose_unit"] == binding.controlled_extension["dose_or_intensity"]["unit"]


@pytest.mark.parametrize("family_uid", FAMILY_UIDS)
def test_delayed_dynamics_are_action_sensitive(config: dict[str, object], family_uid: str) -> None:
    family = _family(config, family_uid)
    action, extension = _alternative_action(family)
    baseline, _ = generate_runtime_instance(
        config,
        family_uid=family_uid,
        repository_root=REPO_ROOT,
        config_file_identity=HASH,
    )
    altered, _ = generate_runtime_instance(
        config,
        family_uid=family_uid,
        repository_root=REPO_ROOT,
        config_file_identity=HASH,
        canonical_action=action,
        action_extension=extension,
    )
    delay = family["world_action"]["response_delay_steps"]

    assert baseline["observations"][:delay] == altered["observations"][:delay]
    assert any(
        baseline["observations"][index] != altered["observations"][index]
        for index in range(delay, len(baseline["observations"]))
    )


@pytest.mark.parametrize("family_uid", FAMILY_UIDS)
def test_registered_action_allows_only_quantitative_intensity_and_duration_variation(
    config: dict[str, object], family_uid: str
) -> None:
    family = _family(config, family_uid)
    action, extension = _alternative_action(family)
    duration_bounds = family["support_domain"]["supported_duration"]
    duration = float(duration_bounds["minimum"]) + 0.8 * (
        float(duration_bounds["maximum"]) - float(duration_bounds["minimum"])
    )
    action["duration"]["value"] = duration
    extension["duration"]["value"] = duration

    binding = bind_world_action(family, action, extension)

    assert binding.mapping_status == "canonical_mapping_validated"
    assert binding.canonical_world_action["duration"]["value"] == duration
    assert binding.controlled_extension["duration"]["value"] == duration


@pytest.mark.parametrize("family_uid", FAMILY_UIDS)
@pytest.mark.parametrize("field", ["dose_or_intensity", "duration"])
def test_registered_action_units_are_immutable(
    config: dict[str, object], family_uid: str, field: str
) -> None:
    family = _family(config, family_uid)
    action = deepcopy(family["world_action"]["canonical"])
    extension = deepcopy(family["world_action"]["extension"])
    extension[field]["unit"] = "forged_unit"
    if field == "duration":
        action["duration"]["unit"] = "forged_unit"
    elif action["intervention_type"] == "pharmacological":
        action["dose_unit"] = "forged_unit"

    with pytest.raises(ControlledWorldContractError) as error:
        bind_world_action(family, action, extension)
    assert _error_code(error) in {"action_mapping_mismatch", "invalid_world_action"}


@pytest.mark.parametrize("family_uid", FAMILY_UIDS)
@pytest.mark.parametrize("mutation", ["target", "agent", "schedule", "program", "provenance"])
def test_paired_registered_action_semantic_forgery_fails_closed(
    config: dict[str, object], family_uid: str, mutation: str
) -> None:
    family = _family(config, family_uid)
    action = deepcopy(family["world_action"]["canonical"])
    extension = deepcopy(family["world_action"]["extension"])
    if mutation == "target":
        action["target_set"] = ["forged_registered_target"]
        extension["action_semantics"]["target_set"] = ["forged_registered_target"]
    elif mutation == "agent":
        action["agent_set"] = ["forged_registered_agent"]
        extension["action_semantics"]["agent_set"] = ["forged_registered_agent"]
    elif mutation == "schedule":
        schedule = {"kind": "program", "program_ref": "forged_registered_program"}
        if action["intervention_type"] == "pharmacological":
            schedule = {"kind": "interval", "interval": 2.0, "unit": "hour"}
        action["schedule"] = schedule
        extension["action_semantics"]["schedule"] = deepcopy(schedule)
    elif mutation == "program":
        program = action["stimulation_or_environment"]
        semantic_field = next(field for field in program if field != "modality")
        program[semantic_field] = "forged_registered_program"
    else:
        action["action_provenance_ref"] = "forged_controlled_world_config"
        extension["action_semantics"]["provenance_ref"] = "forged_controlled_world_config"

    with pytest.raises(ControlledWorldContractError) as error:
        bind_world_action(family, action, extension)
    assert _error_code(error) == "action_mapping_mismatch"


@pytest.mark.parametrize(
    ("family_uid", "parameter", "factor"),
    [
        ("electrophysiology_conductance_world", "stimulation_gain", 0.5),
        ("calcium_synaptic_world", "release_coupling", 0.5),
        ("molecular_pharmacological_causal_world", "binding_affinity", 1.5),
    ],
)
def test_harness_true_utility_depends_on_bound_action_and_family_trajectory(
    config: dict[str, object], family_uid: str, parameter: str, factor: float
) -> None:
    family = _family(config, family_uid)
    oracle = load_harness_oracle(config, family_uid=family_uid, reader_role="benchmark_harness")
    baseline_binding = bind_world_action(family)
    baseline_trajectory = simulate_harness_trajectory(family, oracle, baseline_binding.to_dict())
    baseline_utility = evaluate_harness_true_utility(
        family,
        oracle,
        baseline_binding.to_dict(),
        baseline_trajectory,
        reader_role="benchmark_harness",
    )

    action, extension = _alternative_action(family)
    alternative_binding = bind_world_action(family, action, extension)
    alternative_trajectory = simulate_harness_trajectory(family, oracle, alternative_binding.to_dict())
    alternative_utility = evaluate_harness_true_utility(
        family,
        oracle,
        alternative_binding.to_dict(),
        alternative_trajectory,
        reader_role="benchmark_harness",
    )
    altered_oracle = deepcopy(oracle)
    altered_oracle["oracle_parameters"][parameter] *= factor
    altered_trajectory = simulate_harness_trajectory(family, altered_oracle, baseline_binding.to_dict())
    dynamics_utility = evaluate_harness_true_utility(
        family,
        altered_oracle,
        baseline_binding.to_dict(),
        altered_trajectory,
        reader_role="benchmark_harness",
    )

    assert alternative_utility != baseline_utility
    assert dynamics_utility != baseline_utility

    with pytest.raises(ControlledWorldContractError) as error:
        evaluate_harness_true_utility(
            family,
            oracle,
            alternative_binding.to_dict(),
            baseline_trajectory,
            reader_role="benchmark_harness",
        )
    assert _error_code(error) == "harness_trajectory_mismatch"


def test_harness_true_utility_is_family_specific_and_harness_only(config: dict[str, object]) -> None:
    utilities = {}
    for family_uid in FAMILY_UIDS:
        family = _family(config, family_uid)
        oracle = load_harness_oracle(config, family_uid=family_uid, reader_role="benchmark_harness")
        binding = bind_world_action(family)
        trajectory = simulate_harness_trajectory(family, oracle, binding.to_dict())
        utilities[family_uid] = evaluate_harness_true_utility(
            family,
            oracle,
            binding.to_dict(),
            trajectory,
            reader_role="benchmark_harness",
        )
        with pytest.raises(ControlledWorldContractError) as error:
            evaluate_harness_true_utility(
                family,
                oracle,
                binding.to_dict(),
                trajectory,
                reader_role="operational_twin",
            )
        assert _error_code(error) == "oracle_access_forbidden"
    assert len(set(utilities.values())) == len(FAMILY_UIDS)


@pytest.mark.parametrize("family_uid", FAMILY_UIDS)
def test_information_actions_are_costly_partial_or_failed(config: dict[str, object], family_uid: str) -> None:
    instance = _instance(config, family_uid)
    results = instance["information_results"]

    assert {row["status"] for row in results} == {"partial", "failed"}
    assert all(row["cost"]["value"] > 0 for row in results)
    assert all(row["missing_observables"] for row in results)
    partial = next(row for row in results if row["status"] == "partial")
    failed = next(row for row in results if row["status"] == "failed")
    assert partial["measurements"]
    assert partial["failure_code"] is None
    assert failed["measurements"] == {}
    assert failed["failure_code"].startswith("deterministic_")


def test_unknown_top_level_and_family_fields_fail_closed(config: dict[str, object]) -> None:
    malformed = deepcopy(config)
    malformed["controlled_neural_world"]["silent_fallback"] = True
    with pytest.raises(ControlledWorldContractError) as error:
        validate_controlled_world_config(malformed)
    assert _error_code(error) == "unknown_config_field"

    malformed = deepcopy(config)
    malformed["controlled_neural_world"]["families"][0]["silent_fallback"] = True
    with pytest.raises(ControlledWorldContractError) as error:
        validate_controlled_world_config(malformed)
    assert _error_code(error) == "unknown_config_field"


@pytest.mark.parametrize(
    ("family_uid", "parameter", "replacement", "expected_code"),
    [
        ("electrophysiology_conductance_world", "stimulation_gain", {"value": float("nan"), "unit": "mV_per_uA_ms"}, "non_finite_value"),
        ("calcium_synaptic_world", "release_coupling", {"value": 4.5, "unit": "wrong"}, "invalid_unit"),
        ("molecular_pharmacological_causal_world", "binding_affinity", {"value": float("inf"), "unit": "nmol/L"}, "non_finite_value"),
    ],
)
def test_family_parameter_units_nan_and_inf_fail_closed(
    config: dict[str, object], family_uid: str, parameter: str, replacement: object, expected_code: str
) -> None:
    malformed = deepcopy(config)
    family = select_family_config(malformed["controlled_neural_world"], family_uid)
    family["generator"]["parameters"][parameter] = replacement
    for index, row in enumerate(malformed["controlled_neural_world"]["families"]):
        if row["family_uid"] == family_uid:
            malformed["controlled_neural_world"]["families"][index] = family
    with pytest.raises(ControlledWorldContractError) as error:
        validate_controlled_world_config(malformed)
    assert _error_code(error) == expected_code


@pytest.mark.parametrize("indices", [[999], [-1], [True], [0, 0]])
def test_invalid_observation_indices_fail_typed_and_schema_validation(
    config: dict[str, object], indices: list[object]
) -> None:
    malformed = deepcopy(config)
    malformed["controlled_neural_world"]["families"][0]["world"]["observation_policy"]["observed_indices"]["membrane_voltage"] = indices
    with pytest.raises(ControlledWorldContractError) as error:
        validate_controlled_world_config(malformed)
    assert _error_code(error) == "invalid_observation_index"
    schema = json.loads(WORLD_SCHEMA_PATH.read_text(encoding="utf-8"))
    assert list(Draft202012Validator(schema).iter_errors(malformed["controlled_neural_world"]))


@pytest.mark.parametrize(
    "field",
    ["world_uid", "latent_generator_lineage_uid", "action_uid", "information_action_uid", "output_path", "seed", "subseed"],
)
def test_cross_family_identity_collisions_fail_closed(config: dict[str, object], field: str) -> None:
    malformed = deepcopy(config)
    first, second = malformed["controlled_neural_world"]["families"][:2]
    if field == "world_uid":
        second["world"]["world_uid"] = first["world"]["world_uid"]
    elif field == "latent_generator_lineage_uid":
        second["generator"]["latent_generator_lineage_uid"] = first["generator"]["latent_generator_lineage_uid"]
    elif field == "action_uid":
        second["world_action"]["canonical"]["action_uid"] = first["world_action"]["canonical"]["action_uid"]
        second["information_actions"][0]["canonical"]["candidate_uids"] = [first["world_action"]["canonical"]["action_uid"]]
        second["information_actions"][1]["canonical"]["candidate_uids"] = [first["world_action"]["canonical"]["action_uid"]]
    elif field == "information_action_uid":
        second["information_actions"][0]["canonical"]["information_action_uid"] = first["information_actions"][0]["canonical"]["information_action_uid"]
    elif field == "output_path":
        second["output_paths"]["world_instance"] = first["output_paths"]["world_instance"]
    elif field == "seed":
        second["generator"]["seed"] = first["generator"]["seed"]
    else:
        second["generator"]["subseeds"]["parameter_jitter"] = first["generator"]["subseeds"]["parameter_jitter"]
    with pytest.raises(ControlledWorldContractError) as error:
        validate_controlled_world_config(malformed)
    assert _error_code(error) in {"duplicate_family_identity", "duplicate_seed_identity"}


def test_missing_or_extra_family_fails_closed(config: dict[str, object]) -> None:
    missing = deepcopy(config)
    missing["controlled_neural_world"]["families"].pop()
    with pytest.raises(ControlledWorldContractError) as error:
        validate_controlled_world_config(missing)
    assert _error_code(error) == "unsupported_world_family"

    duplicate = deepcopy(config)
    duplicate["controlled_neural_world"]["families"].append(deepcopy(duplicate["controlled_neural_world"]["families"][0]))
    with pytest.raises(ControlledWorldContractError) as error:
        validate_controlled_world_config(duplicate)
    assert _error_code(error) in {"unsupported_world_family", "duplicate_family_identity"}


@pytest.mark.parametrize("family_uid", FAMILY_UIDS)
def test_unsupported_and_forced_ood_actions_fail_closed(config: dict[str, object], family_uid: str) -> None:
    family = _family(config, family_uid)
    action = deepcopy(family["world_action"]["canonical"])
    extension = deepcopy(family["world_action"]["extension"])
    if family_uid == "molecular_pharmacological_causal_world":
        action["route"] = "oral"
        extension["action_semantics"]["route"] = "oral"
    else:
        action["intervention_type"] = "genetic"
        action["stimulation_or_environment"]["modality"] = "genetic"
        extension["modality"] = "genetic"
    with pytest.raises(ControlledWorldContractError) as error:
        generate_runtime_instance(
            config,
            family_uid=family_uid,
            repository_root=REPO_ROOT,
            config_file_identity=HASH,
            canonical_action=action,
            action_extension=extension,
        )
    assert _error_code(error) == "action_mapping_mismatch"

    extension = deepcopy(family["world_action"]["extension"])
    extension["dose_or_intensity"]["value"] = float(family["support_domain"]["supported_intensity"]["maximum"]) + 100.0
    action = deepcopy(family["world_action"]["canonical"])
    if action["intervention_type"] == "pharmacological":
        action["dose"] = extension["dose_or_intensity"]["value"]
    with pytest.raises(ControlledWorldContractError) as error:
        generate_runtime_instance(
            config,
            family_uid=family_uid,
            repository_root=REPO_ROOT,
            config_file_identity=HASH,
            canonical_action=action,
            action_extension=extension,
            force_execute=True,
        )
    assert _error_code(error) == "forced_ood_execution"


def test_cross_family_world_action_reuse_fails_closed(config: dict[str, object]) -> None:
    electrophysiology = _family(config, FAMILY_UIDS[0])
    calcium = _family(config, FAMILY_UIDS[1])
    with pytest.raises(ControlledWorldContractError) as error:
        bind_world_action(
            calcium,
            electrophysiology["world_action"]["canonical"],
            electrophysiology["world_action"]["extension"],
        )
    assert _error_code(error) == "cross_family_action_reuse"


@pytest.mark.parametrize("field", ["dose", "dose_unit", "route", "schedule", "target_set", "agent_set"])
def test_pharmacological_action_mapping_is_exact(config: dict[str, object], field: str) -> None:
    family = _family(config, "molecular_pharmacological_causal_world")
    action = deepcopy(family["world_action"]["canonical"])
    extension = deepcopy(family["world_action"]["extension"])
    if field == "dose":
        action["dose"] = 3.0
    elif field == "dose_unit":
        action["dose_unit"] = "ug/kg"
    elif field == "route":
        action["route"] = "intravenous"
    elif field == "schedule":
        action["schedule"] = {"kind": "single"}
    elif field == "target_set":
        action["target_set"] = ["different_target"]
    else:
        action["agent_set"] = ["different_agent"]
    with pytest.raises(ControlledWorldContractError) as error:
        bind_world_action(family, action, extension)
    assert _error_code(error) == "action_mapping_mismatch"


def test_action_context_drift_is_explicit_ood(config: dict[str, object]) -> None:
    family = _family(config, "calcium_synaptic_world")
    action = deepcopy(family["world_action"]["canonical"])
    extension = deepcopy(family["world_action"]["extension"])
    extension["biological_context"]["region"] = "entorhinal_cortex"
    action["state_scope"]["region"] = ["entorhinal_cortex"]
    with pytest.raises(ControlledWorldContractError) as error:
        generate_runtime_instance(
            config,
            family_uid="calcium_synaptic_world",
            repository_root=REPO_ROOT,
            config_file_identity=HASH,
            canonical_action=action,
            action_extension=extension,
        )
    assert _error_code(error) == "action_mapping_mismatch"


def test_unsupported_and_hidden_information_requests_fail_closed(config: dict[str, object]) -> None:
    family = _family(config, "calcium_synaptic_world")
    spec = deepcopy(family["information_actions"][0])
    spec["canonical"]["requested_observables"] = ["synaptic_weight"]
    spec["outcome_policy"] = {
        "status": "failed",
        "measured_observables": [],
        "missing_observables": ["synaptic_weight"],
        "noise_offsets": {},
        "failure_code": "unsupported",
    }
    with pytest.raises(ControlledWorldContractError) as error:
        bind_information_action(family, spec)
    assert _error_code(error) == "unsupported_information_request"

    for requested in ("hidden_optimum", "true_utility", "oracle_parameters"):
        spec = deepcopy(family["information_actions"][0])
        spec["canonical"]["requested_observables"] = [requested]
        spec["outcome_policy"] = {
            "status": "failed",
            "measured_observables": [],
            "missing_observables": [requested],
            "noise_offsets": {},
            "failure_code": "forbidden",
        }
        with pytest.raises(ControlledWorldContractError) as error:
            bind_information_action(family, spec)
        assert _error_code(error) in {"hidden_optimum_request_forbidden", "oracle_access_forbidden"}


def test_cross_family_information_context_and_candidate_fail_closed(config: dict[str, object]) -> None:
    family = _family(config, "calcium_synaptic_world")
    spec = deepcopy(family["information_actions"][0])
    spec["canonical"]["candidate_uids"] = ["electrophysiology_stimulation_action"]
    with pytest.raises(ControlledWorldContractError) as error:
        bind_information_action(family, spec)
    assert _error_code(error) == "cross_family_action_reuse"

    spec = deepcopy(family["information_actions"][0])
    spec["canonical"]["measurement_context"]["region"] = "hippocampus_ca1"
    with pytest.raises(ControlledWorldContractError) as error:
        bind_information_action(family, spec)
    assert _error_code(error) == "information_context_mismatch"


@pytest.mark.parametrize("family_uid", FAMILY_UIDS)
def test_non_harness_reader_cannot_load_oracle(config: dict[str, object], family_uid: str) -> None:
    with pytest.raises(ControlledWorldContractError) as error:
        load_harness_oracle(config, family_uid=family_uid, reader_role="operational_twin")
    assert _error_code(error) == "oracle_access_forbidden"


@pytest.mark.parametrize("family_uid", FAMILY_UIDS)
def test_oracle_protected_and_access_injection_fail_closed(config: dict[str, object], family_uid: str) -> None:
    instance = _instance(config, family_uid)
    instance["hidden_state"] = {"leak": 0.1}
    with pytest.raises(ControlledWorldContractError) as error:
        validate_runtime_instance(instance)
    assert _error_code(error) == "oracle_access_forbidden"

    family = _family(config, family_uid)
    action = deepcopy(family["world_action"]["canonical"])
    action["action_provenance_ref"] = "protected_holdout/result.json"
    extension = deepcopy(family["world_action"]["extension"])
    extension["action_semantics"]["provenance_ref"] = "protected_holdout/result.json"
    with pytest.raises(ControlledWorldContractError) as error:
        bind_world_action(family, action, extension)
    assert _error_code(error) == "protected_reference_forbidden"

    instance = _instance(config, family_uid)
    instance["access_counters"]["six_context_outcome_access"] = 1
    with pytest.raises(ControlledWorldContractError) as error:
        validate_runtime_instance(instance)
    assert _error_code(error) == "protected_access_nonzero"


@pytest.mark.parametrize("family_uid", FAMILY_UIDS)
def test_config_aware_runtime_rejects_semantic_forgery(config: dict[str, object], family_uid: str) -> None:
    instance = _instance(config, family_uid)
    instance["generator_provenance"]["seed_identity"] = "b" * 64
    with pytest.raises(ControlledWorldContractError) as error:
        validate_runtime_instance(instance, config=config, repository_root=REPO_ROOT)
    assert _error_code(error) == "runtime_config_mismatch"
    assert "generator_provenance" in error.value.details["fields"]


@pytest.mark.parametrize("family_uid", FAMILY_UIDS)
def test_contradictory_information_result_status_fails_closed(config: dict[str, object], family_uid: str) -> None:
    instance = _instance(config, family_uid)
    partial = next(row for row in instance["information_results"] if row["status"] == "partial")
    partial["measurements"] = {}
    with pytest.raises(ControlledWorldContractError) as error:
        validate_runtime_instance(instance, config=config, repository_root=REPO_ROOT)
    assert _error_code(error) == "invalid_information_outcome"


@pytest.mark.parametrize("family_uid", FAMILY_UIDS)
def test_replay_identity_unknown_field_and_oracle_injection_fail_closed(
    config: dict[str, object], family_uid: str
) -> None:
    _, manifest = generate_runtime_and_manifest(
        config,
        family_uid=family_uid,
        repository_root=REPO_ROOT,
        config_file_identity=CONFIG_FILE_IDENTITY,
    )
    tampered = deepcopy(manifest)
    tampered["observable_output_identity"] = "b" * 64
    with pytest.raises(ControlledWorldContractError) as error:
        replay_runtime_instance(config, tampered, repository_root=REPO_ROOT)
    assert _error_code(error) == "replay_identity_mismatch"

    tampered = deepcopy(manifest)
    tampered["silent_fallback"] = True
    with pytest.raises(ControlledWorldContractError) as error:
        replay_runtime_instance(config, tampered, repository_root=REPO_ROOT)
    assert _error_code(error) == "unknown_replay_manifest_field"

    tampered = deepcopy(manifest)
    tampered["oracle_state"] = {"hidden_optimum": 1.0}
    with pytest.raises(ControlledWorldContractError) as error:
        replay_runtime_instance(config, tampered, repository_root=REPO_ROOT)
    assert _error_code(error) == "oracle_access_forbidden"


@pytest.mark.parametrize(
    ("path", "replacement", "mismatch_field"),
    [
        (("replay_uid",), "controlled_world_replay_forged", "replay_uid"),
        (("generator_implementation_uid",), "forged_generator", "generator_implementation_uid"),
        (("input_lineage", 0, "artifact_ref"), "Libs/utils/data/forged_runtime.py", "input_lineage"),
        (("output_lineage", 0, "content_identity"), "b" * 64, "output_lineage"),
    ],
)
def test_every_replay_field_is_authoritatively_compared(
    config: dict[str, object], path: tuple[object, ...], replacement: object, mismatch_field: str
) -> None:
    _, manifest = generate_runtime_and_manifest(
        config,
        family_uid="calcium_synaptic_world",
        repository_root=REPO_ROOT,
        config_file_identity=CONFIG_FILE_IDENTITY,
    )
    _set_nested(manifest, path, replacement)
    with pytest.raises(ControlledWorldContractError) as error:
        replay_runtime_instance(config, manifest, repository_root=REPO_ROOT)
    assert _error_code(error) == "replay_identity_mismatch"
    assert mismatch_field in error.value.details["fields"]


def test_replay_reproduction_contract_mutation_fails_closed(config: dict[str, object]) -> None:
    _, manifest = generate_runtime_and_manifest(
        config,
        family_uid="molecular_pharmacological_causal_world",
        repository_root=REPO_ROOT,
        config_file_identity=CONFIG_FILE_IDENTITY,
    )
    manifest["deterministic_reproduction"]["replay_count"] = 2
    with pytest.raises(ControlledWorldContractError) as error:
        replay_runtime_instance(config, manifest, repository_root=REPO_ROOT)
    assert _error_code(error) == "invalid_replay_contract"


@pytest.mark.parametrize(
    ("field", "replacement", "expected_code"),
    [
        ("benchmark_split_assignment", "frozen", "benchmark_scope_violation"),
        ("sealed_test_partitions", "created", "benchmark_scope_violation"),
        ("information_headroom_qualification", "passed", "benchmark_scope_violation"),
        ("benchmark_qualification", "CONTROLLED_NEURAL_WORLD_BENCHMARK_QUALIFIED", "benchmark_scope_violation"),
    ],
)
def test_family_manifest_cannot_claim_freeze_or_qualification(
    config: dict[str, object], tmp_path: Path, field: str, replacement: str, expected_code: str
) -> None:
    paths = build_controlled_neural_world(config_path=CONFIG_PATH, repository_root=REPO_ROOT, artifact_root=tmp_path)
    packages = _packages(config)
    manifest = json.loads(paths["family_manifest"].read_text(encoding="utf-8"))
    manifest[field] = replacement
    with pytest.raises(ControlledWorldContractError) as error:
        validate_family_manifest(
            manifest,
            config=config,
            config_file_identity=CONFIG_FILE_IDENTITY,
            authoritative_packages=packages,
        )
    assert _error_code(error) == expected_code


def test_family_manifest_omission_duplicate_and_hash_tamper_fail_closed(
    config: dict[str, object], tmp_path: Path
) -> None:
    paths = build_controlled_neural_world(config_path=CONFIG_PATH, repository_root=REPO_ROOT, artifact_root=tmp_path)
    packages = _packages(config)
    manifest = json.loads(paths["family_manifest"].read_text(encoding="utf-8"))
    omitted = deepcopy(manifest)
    omitted["families"].pop()
    with pytest.raises(ControlledWorldContractError) as error:
        validate_family_manifest(
            omitted,
            config=config,
            config_file_identity=CONFIG_FILE_IDENTITY,
            authoritative_packages=packages,
        )
    assert _error_code(error) == "family_manifest_incomplete"

    duplicate = deepcopy(manifest)
    duplicate["families"][1]["world_uid"] = duplicate["families"][0]["world_uid"]
    with pytest.raises(ControlledWorldContractError) as error:
        validate_family_manifest(
            duplicate,
            config=config,
            config_file_identity=CONFIG_FILE_IDENTITY,
            authoritative_packages=packages,
        )
    assert _error_code(error) == "duplicate_family_identity"

    manifest["families"][0]["runtime_content_identity"] = "b" * 64
    paths["family_manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = validate_controlled_neural_world(
        config_path=CONFIG_PATH,
        repository_root=REPO_ROOT,
        artifact_root=tmp_path,
    )
    assert audit["status"] == "blocked"
    assert {row["code"] for row in audit["blockers"]} & {
        "family_manifest_identity_mismatch",
        "aggregate_hash_mismatch",
    }


@pytest.mark.parametrize("field", ["manifest_uid", "source_config_identity"])
def test_family_manifest_authority_identity_tamper_fails_closed(
    config: dict[str, object], tmp_path: Path, field: str
) -> None:
    paths = build_controlled_neural_world(config_path=CONFIG_PATH, repository_root=REPO_ROOT, artifact_root=tmp_path)
    packages = _packages(config)
    manifest = json.loads(paths["family_manifest"].read_text(encoding="utf-8"))
    manifest[field] = "b" * 64
    with pytest.raises(ControlledWorldContractError) as error:
        validate_family_manifest(
            manifest,
            config=config,
            config_file_identity=CONFIG_FILE_IDENTITY,
            authoritative_packages=packages,
        )
    assert _error_code(error) in {
        "family_manifest_identity_mismatch",
        "family_manifest_config_mismatch",
    }


def test_family_manifest_source_config_reference_is_authoritative(
    config: dict[str, object], tmp_path: Path
) -> None:
    paths = build_controlled_neural_world(config_path=CONFIG_PATH, repository_root=REPO_ROOT, artifact_root=tmp_path)
    packages = _packages(config)
    manifest = json.loads(paths["family_manifest"].read_text(encoding="utf-8"))
    manifest["source_config_ref"] = "Input/forged_controlled_neural_world.yaml"

    with pytest.raises(ControlledWorldContractError) as error:
        validate_family_manifest(
            manifest,
            config=config,
            config_file_identity=CONFIG_FILE_IDENTITY,
            authoritative_packages=packages,
        )
    assert _error_code(error) == "family_manifest_config_mismatch"


@pytest.mark.parametrize(
    ("field", "replacement", "expected_code"),
    [
        ("family_uid", "forged_world_family", "unsupported_world_family"),
        ("world_uid", "forged_world_uid", "family_manifest_authority_mismatch"),
        ("latent_generator_lineage_uid", "forged_latent_lineage", "family_manifest_authority_mismatch"),
        ("seed", 99173, "family_manifest_authority_mismatch"),
        ("seed_identity", "b" * 64, "family_manifest_authority_mismatch"),
        (
            "runtime_artifact_ref",
            "Input/processed/controlled_neural_world/forged_runtime.json",
            "family_manifest_authority_mismatch",
        ),
        ("runtime_content_identity", "c" * 64, "family_manifest_authority_mismatch"),
        (
            "replay_artifact_ref",
            "Input/processed/controlled_neural_world/forged_replay.json",
            "family_manifest_authority_mismatch",
        ),
        ("replay_content_identity", "d" * 64, "family_manifest_authority_mismatch"),
        ("replay_uid", "controlled_world_replay_forged", "family_manifest_authority_mismatch"),
    ],
)
def test_fully_rehashed_family_manifest_lineage_forgery_fails_closed(
    config: dict[str, object], tmp_path: Path, field: str, replacement: object, expected_code: str
) -> None:
    paths = build_controlled_neural_world(
        config_path=CONFIG_PATH,
        repository_root=REPO_ROOT,
        artifact_root=tmp_path,
    )
    packages = _packages(config)
    manifest = json.loads(paths["family_manifest"].read_text(encoding="utf-8"))
    manifest["families"][0][field] = replacement
    _rehash_family_manifest(manifest)

    with pytest.raises(ControlledWorldContractError) as error:
        validate_family_manifest(
            manifest,
            config=config,
            config_file_identity=CONFIG_FILE_IDENTITY,
            authoritative_packages=packages,
        )
    assert _error_code(error) == expected_code
    if expected_code == "family_manifest_authority_mismatch":
        assert field in error.value.details["fields"]

    paths["family_manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit = validate_controlled_neural_world(
        config_path=CONFIG_PATH,
        repository_root=REPO_ROOT,
        artifact_root=tmp_path,
    )
    assert audit["status"] == "blocked"
    if expected_code == "family_manifest_authority_mismatch":
        authority_blocker = next(
            row for row in audit["blockers"] if row["code"] == "family_manifest_authority_mismatch"
        )
        assert field in authority_blocker["fields"]
    else:
        assert expected_code in {row["code"] for row in audit["blockers"]}


def test_validator_blocks_schema_valid_tampered_runtime_and_replay(tmp_path: Path) -> None:
    paths = build_controlled_neural_world(config_path=CONFIG_PATH, repository_root=REPO_ROOT, artifact_root=tmp_path)
    runtime_path = paths["calcium_synaptic_world.world_instance"]
    instance = json.loads(runtime_path.read_text(encoding="utf-8"))
    instance["generator_provenance"]["implementation_uid"] = "schema_valid_forged_generator"
    runtime_path.write_text(json.dumps(instance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    replay_path = paths["molecular_pharmacological_causal_world.replay_manifest"]
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    replay["generator_implementation_uid"] = "schema_valid_forged_generator"
    replay_path.write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    audit = validate_controlled_neural_world(
        config_path=CONFIG_PATH,
        repository_root=REPO_ROOT,
        artifact_root=tmp_path,
    )
    codes = {row["code"] for row in audit["blockers"]}
    assert audit["status"] == "blocked"
    assert "runtime_config_mismatch" in codes
    assert "replay_identity_mismatch" in codes


def test_schema_rejects_unknown_runtime_and_manifest_fields(config: dict[str, object], tmp_path: Path) -> None:
    schema = json.loads(WORLD_SCHEMA_PATH.read_text(encoding="utf-8"))
    instance = _instance(config, "electrophysiology_conductance_world")
    instance["future_observations"] = []
    assert list(Draft202012Validator(schema).iter_errors(instance))

    paths = build_controlled_neural_world(config_path=CONFIG_PATH, repository_root=REPO_ROOT, artifact_root=tmp_path)
    manifest = json.loads(paths["family_manifest"].read_text(encoding="utf-8"))
    manifest["qualification_override"] = True
    assert list(Draft202012Validator(schema).iter_errors(manifest))
