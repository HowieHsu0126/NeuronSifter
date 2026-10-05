from __future__ import annotations

import copy
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ROOT = REPO_ROOT / "Libs" / "configs" / "exp"


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
            formal_seed_values = (
                fallback_config.get("screening_pipeline", {})
                .get("screening_benchmark", {})
                .get("formal_seed_values", [])
            )
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
    formal_seed_values = (
        fallback_config.get("screening_pipeline", {})
        .get("screening_benchmark", {})
        .get("formal_seed_values", [])
    )
    runtime_payload = merged_payload.get("runtime", {})
    if formal_seed_values and isinstance(runtime_payload, dict) and "seed_values" in runtime_payload:
        runtime_payload["seed_values"] = [int(seed_value) for seed_value in formal_seed_values]
    merged_payload.setdefault("family_name", profile_name)
    merged_payload.setdefault("paper_track", "")
    return merged_payload


def load_paper_track(track_name: str) -> dict[str, object]:
    return load_named_manifest("paper_tracks", track_name)
