from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.reports.build_paper_results import canonical_dynamics_spec, required_dynamics_profiles
from Libs.utils.exp.experiment_manifests import load_experiment_family, load_paper_track
from Libs.utils.data.raw_data_registry import load_config


def test_load_experiment_family_resolves_manifest_layers() -> None:
    config = load_config(str(REPO_ROOT / "Libs/configs/config_stack.yaml"))
    profile = load_experiment_family("dynamics_mainline_ad", config)
    assert profile["family_name"] == "dynamics_mainline_ad"
    assert profile["paper_track"] == "research_hardening"
    assert profile["disease_program"] == "ad_control"
    assert profile["model_profile"] == "dynamics_twin"
    assert profile["runtime"]["benchmark_name_template"] == "dynamics_twin_ad_mainline_seed_{seed}"
    assert profile["execution"]["prebuild_commands"]
    assert profile["execution"]["train_commands"]


def test_rollout_operator_without_mechanism_experts_resolves_train_commands() -> None:
    config = load_config(str(REPO_ROOT / "Libs/configs/config_stack.yaml"))
    profile = load_experiment_family("rollout_operator_without_mechanism_experts", config)
    assert profile["family_name"] == "rollout_operator_without_mechanism_experts"
    assert profile["paper_track"] == "research_hardening"
    assert profile["disease_program"] == "ad_control"
    assert profile["model_profile"] == "dynamics_twin"
    assert profile["execution"]["train_commands"]


def test_load_paper_track_lists_primary_dynamics_profiles() -> None:
    track = load_paper_track("research_hardening")
    assert "dynamics_mainline_ad" in track["primary_dynamics_profiles"]
    assert "shared_latent_formal" in track["shared_context_profiles"]


def test_paper_track_canonical_dynamics_mapping_excludes_context_profiles_from_required_set() -> None:
    track = load_paper_track("research_hardening")
    dynamics_spec = canonical_dynamics_spec(track)
    assert dynamics_spec["ad_mainline_profile"] == "dynamics_mainline_ad"
    assert dynamics_spec["context_formal_profiles"] == []
    required_profiles = required_dynamics_profiles(track)
    assert "dynamics_mainline_ad" in required_profiles
    assert "shared_latent_formal" not in required_profiles


def test_dynamics_twin_prebuild_materializes_foundation_registries() -> None:
    config = load_config(str(REPO_ROOT / "Libs/configs/config_stack.yaml"))
    profile = load_experiment_family("dynamics_mainline_ad", config)
    prebuild_commands = " ".join(profile["execution"]["prebuild_commands"])
    for command_name in [
        "build_world_state_ready_registry.py",
        "validate_world_state_ready_registry.py",
        "build_measurement_semantics_registry.py",
        "validate_measurement_semantics_registry.py",
        "build_crossmodal_anchor_ready_registry.py",
        "validate_crossmodal_anchor_ready_registry.py",
        "build_simulation_teacher_ready_registry.py",
        "validate_simulation_teacher_ready_registry.py",
        "validate_mechanistic_teacher_runtime.py",
        "build_counterfactual_benchmark_ready_registry.py",
        "validate_counterfactual_benchmark_ready_registry.py",
        "build_feedback_ready_registry.py",
        "validate_feedback_ready_registry.py",
    ]:
        assert command_name in prebuild_commands


def test_pd_partial_public_profile_resolves_through_manifest_layers() -> None:
    config = load_config(str(REPO_ROOT / "Libs/configs/config_stack.yaml"))
    profile = load_experiment_family("dynamics_mainline_pd", config)
    assert profile["family_name"] == "dynamics_mainline_pd"
    assert profile["disease_program"] == "pd_control"
    assert profile["runtime"]["seed_values"] == [3407, 42, 26]
