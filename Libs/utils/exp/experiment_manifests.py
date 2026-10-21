from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = REPO_ROOT / "Libs" / "configs" / "exp"
NATIVE_SEED_POLICIES = {"manuscript_native_three": [41, 42, 43],
                        "manuscript_native_five": [40, 41, 42, 43, 44]}


def native_training_seed_values(policy: str, values: list[int]) -> list[int]:
    """Validate an explicit active or historical native repetition declaration."""
    if (policy not in NATIVE_SEED_POLICIES or not isinstance(values, list)
            or any(type(value) is not int for value in values)
            or values != NATIVE_SEED_POLICIES[policy]):
        raise ValueError("Native training seed policy requires its exact ordered integer seeds: " + policy)
    return list(values)


def deep_merge(base_payload: dict[str, object], override_payload: dict[str, object]) -> dict[str, object]:
    merged = copy.deepcopy(base_payload)
    for key, value in override_payload.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
            continue
        merged[key] = copy.deepcopy(value)
    return merged


def manifest_path(category: str, name: str) -> Path:
    return EXPERIMENT_ROOT / category / f"{name}.yaml"


def load_yaml_file(path: Path) -> dict[str, object]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_named_manifest(category: str, name: str) -> dict[str, object]:
    path = manifest_path(category, name)
    if not path.exists():
        raise FileNotFoundError(f"Missing experiment manifest: {path}")
    payload = load_yaml_file(path)
    payload.setdefault("manifest_name", name)
    payload.setdefault("manifest_category", category)
    payload.setdefault("manifest_path", str(path.resolve().relative_to(REPO_ROOT)))
    return payload


def resolve_training_seed(config: dict, requested_seed: int | None = None, *,
                          require_selected: bool = False, materialize: bool = False,
                          legacy_seed_values: list[int] | None = None) -> dict:
    """Resolve the prospective training binding without changing any RNG namespace."""
    benchmark = config.get("screening_pipeline", {}).get("screening_benchmark", {})
    training = benchmark.get("training", {})
    policy = training.get("seed_policy", "legacy_formal")
    if policy not in {"legacy_formal", *NATIVE_SEED_POLICIES}:
        raise ValueError(f"Unknown training seed_policy: {policy}")
    selected = config.get("selected_formal_seed")
    effective = training.get("random_seed")
    suite = config.get("paper_grade_full_suite", {})
    if policy in NATIVE_SEED_POLICIES:
        allowed = suite.get("formal_seed_values")
        allowed = native_training_seed_values(policy, allowed)
        for label, values in (("suite.seeds", suite.get("seeds")),
                              ("baseline.seed_values", config.get("paper_grade_baseline_registry", {}).get("seed_values", allowed))):
            if (not isinstance(values, list) or any(type(value) is not int for value in values)
                    or values != allowed):
                raise ValueError(f"Prospective training seed declaration mismatch: {label}")
        for label, value in (("requested", requested_seed), ("selected_formal_seed", selected)):
            if value is not None and (type(value) is not int or value not in allowed):
                raise ValueError(f"Invalid {label} training seed {value!r}; allowed {allowed}")
        if selected is not None:
            if type(effective) is not int or effective != selected:
                raise ValueError("selected_formal_seed differs from effective YAML training seed")
            if requested_seed is not None and requested_seed != selected:
                raise ValueError("Requested training seed differs from selected/effective YAML seed")
        elif materialize and requested_seed is not None:
            selected = requested_seed
        elif require_selected:
            raise ValueError("Prospective training requires a materialized selected_formal_seed; no first-seed default")
        elif requested_seed is not None:
            raise ValueError("Select a prospective training seed through configuration materialization")
        base = selected
        spec = config.get("schemas", {}).get("native_training_seed_binding", {})
        version = spec.get("schema_version")
        if version != "native_training_seed_binding.v1":
            raise ValueError("Missing current native training-seed binding schema")
    else:
        allowed = list(legacy_seed_values if legacy_seed_values is not None else benchmark.get("formal_seed_values", []))
        allowed = [int(value) for value in allowed]
        if requested_seed is not None:
            base = int(requested_seed)
        elif selected is not None:
            base = int(selected)
        elif legacy_seed_values:
            base = int(legacy_seed_values[0])
        elif effective is not None:
            base = int(effective)
        else:
            base = allowed[0] if allowed else None
        if materialize and requested_seed is not None:
            selected = base
        version = "native_training_seed_binding.v1"
    binding = {"schema_version": version, "seed_policy": policy,
               "allowed_training_seeds": list(allowed), "selected_formal_seed": selected,
               "base_training_seed": base, "model_seed": base, "scope": []}
    if policy in NATIVE_SEED_POLICIES and set(binding) != set(spec.get("required_fields", [])):
        raise ValueError("Native training-seed binding schema fields mismatch")
    return binding


def validate_training_seed_checkpoint(payload: dict, expected: dict) -> None:
    """Validate resume/reuse metadata; explicit weight transfer is a separate path."""
    stored = payload.get("training_seed_binding")
    prospective = expected["seed_policy"] in NATIVE_SEED_POLICIES
    if not prospective and not (isinstance(stored, dict) and stored.get("seed_policy") in NATIVE_SEED_POLICIES):
        return
    # JSON's integer/float distinction also rejects silently retagged numeric types.
    def matches(value):
        try:
            return json.dumps(value, sort_keys=True, allow_nan=False) == json.dumps(expected, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError):
            return False
    if not matches(stored):
        raise ValueError("Prospective checkpoint training seed binding mismatch or missing")
    if "best_state" in payload and not matches(payload.get("best_training_seed_binding")):
        raise ValueError("Prospective best checkpoint training seed binding mismatch or missing")


def load_experiment_family(profile_name: str, fallback_config: dict[str, object] | None = None) -> dict[str, object]:
    family_path = manifest_path("experiment_families", profile_name)
    if family_path.exists():
        family_payload = load_named_manifest("experiment_families", profile_name)
        disease_program_name = str(family_payload.get("disease_program", "")).strip()
        model_profile_name = str(family_payload.get("model_profile", "")).strip()
        merged_payload: dict[str, object] = {}
        if disease_program_name:
            merged_payload = deep_merge(merged_payload, load_named_manifest("disease_programs", disease_program_name))
        if model_profile_name:
            merged_payload = deep_merge(merged_payload, load_named_manifest("model_profiles", model_profile_name))
        if fallback_config is not None:
            central_payload = fallback_config.get("paper_experiments", {}).get(profile_name, {})
            merged_payload = deep_merge(merged_payload, central_payload)
        family_identity_payload = copy.deepcopy(family_payload)
        for central_only_key in ["runtime", "cluster", "cluster_prebuild", "cluster_training"]:
            family_identity_payload.pop(central_only_key, None)
        merged_payload = deep_merge(merged_payload, family_identity_payload)
        if fallback_config is not None:
            effective_config = deep_merge(fallback_config, merged_payload.get("config_overrides", {}))
            formal_seed_values = resolve_training_seed(effective_config)["allowed_training_seeds"]
            runtime_payload = merged_payload.get("runtime", {})
            if formal_seed_values and isinstance(runtime_payload, dict) and "seed_values" in runtime_payload:
                runtime_payload["seed_values"] = [int(seed_value) for seed_value in formal_seed_values]
        merged_payload.setdefault("family_name", profile_name)
        return merged_payload
    if fallback_config is None:
        raise FileNotFoundError(f"Missing experiment family manifest: {family_path}")
    paper_experiments = fallback_config.get("paper_experiments", {})
    if profile_name not in paper_experiments:
        raise ValueError(f"Unknown paper experiment profile: {profile_name}")
    merged_payload = copy.deepcopy(paper_experiments[profile_name])
    effective_config = deep_merge(fallback_config, merged_payload.get("config_overrides", {}))
    formal_seed_values = resolve_training_seed(effective_config)["allowed_training_seeds"]
    runtime_payload = merged_payload.get("runtime", {})
    if formal_seed_values and isinstance(runtime_payload, dict) and "seed_values" in runtime_payload:
        runtime_payload["seed_values"] = [int(seed_value) for seed_value in formal_seed_values]
    merged_payload.setdefault("family_name", profile_name)
    merged_payload.setdefault("paper_track", "")
    return merged_payload


def load_paper_track(track_name: str) -> dict[str, object]:
    return load_named_manifest("paper_tracks", track_name)
