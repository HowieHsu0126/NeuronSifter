from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from Libs.utils.data.controlled_world_contracts import FAMILY_UIDS, ControlledWorldContractError
from Libs.utils.data.controlled_world_open_package import (
    OPEN_PACKAGE_POSITIVE_TERMINAL,
    build_controlled_world_open_package,
    build_package_receipt,
    build_open_manifests,
    reject_forbidden_path_tokens,
    validate_open_manifests,
    validate_open_package_config,
)
from Libs.utils.data.raw_data_registry import load_config

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "Libs/configs/config_stack.yaml"


def _small_config() -> dict[str, object]:
    config = load_config(CONFIG_PATH)
    open_cfg = config["controlled_world_open_action_identification"]
    open_cfg["latent_worlds_per_family"] = {
        "training": 1,
        "development": 1,
        "calibration": 1,
    }
    return config


def test_open_package_builds_two_supported_actions_per_family_and_role(tmp_path: Path) -> None:
    config = _small_config()
    paths = build_controlled_world_open_package(
        config=config,
        repository_root=ROOT,
        artifact_root=tmp_path,
    )
    manifests = build_open_manifests(config, repository_root=ROOT)
    receipt = build_package_receipt(manifests, config)
    assert paths["training_manifest"].is_file()
    assert receipt["terminal_decision"] == OPEN_PACKAGE_POSITIVE_TERMINAL
    for role, manifest in manifests.items():
        assert manifest["latent_world_count"] == 3
        assert manifest["supported_execution_count"] == 6
        assert {row["family_uid"] for row in manifest["worlds"]} == set(FAMILY_UIDS)
        for pair in manifest["counterfactual_pairs"]:
            members = [
                row for row in manifest["worlds"]
                if row["counterfactual_pair_uid"] == pair["counterfactual_pair_uid"]
            ]
            assert len(members) == 2
            assert len({row["action_program_uid"] for row in members}) == 2
            assert len({row["observable_output_identity"] for row in members}) == 2
    assert {
        (row["family_uid"], row["support_status"]["status"])
        for row in manifests["development"]["development_support_cases"]
    } == {
        (family_uid, status)
        for family_uid in FAMILY_UIDS
        for status in ("supported", "out_of_distribution", "unsupported")
    }


def test_open_package_validator_rejects_single_or_duplicate_pair_membership() -> None:
    config = _small_config()
    manifests = build_open_manifests(config, repository_root=ROOT)
    broken = deepcopy(manifests)
    pair_uid = broken["training"]["counterfactual_pairs"][0]["counterfactual_pair_uid"]
    broken["training"]["worlds"][2]["counterfactual_pair_uid"] = pair_uid
    blockers = validate_open_manifests(broken, config=config, repository_root=ROOT, replay=False)
    assert {row["code"] for row in blockers} & {
        "counterfactual_pair_cardinality_invalid",
        "pair_authority_membership_mismatch",
    }


def test_open_package_validator_rejects_cross_role_latent_overlap() -> None:
    config = _small_config()
    manifests = build_open_manifests(config, repository_root=ROOT)
    manifests["development"]["worlds"][0]["latent_world_uid"] = manifests["training"]["worlds"][0]["latent_world_uid"]
    blockers = validate_open_manifests(manifests, config=config, repository_root=ROOT, replay=False)
    assert "cross_role_latent_overlap" in {row["code"] for row in blockers}


def test_open_package_validator_rejects_nonzero_access_and_open_gate() -> None:
    config = _small_config()
    bad_access = deepcopy(config)
    bad_access["controlled_world_open_action_identification"]["access_policy"]["protected_holdout_access"] = 1
    with pytest.raises(ControlledWorldContractError, match="zero"):
        validate_open_package_config(bad_access)
    bad_gate = deepcopy(config)
    bad_gate["controlled_world_open_action_identification"]["closed_downstream_flags"]["training_authorized"] = True
    with pytest.raises(ControlledWorldContractError, match="gates"):
        validate_open_package_config(bad_gate)


def test_forbidden_path_tokens_are_rejected_before_access() -> None:
    for path in (
        "Input/processed/controlled_neural_world/benchmark/test_manifest.json",
        "Input/processed/controlled_neural_world/benchmark/access_receipts/twin_qualification_unopened_access_receipt.json",
        "Input/processed/controlled_neural_world/protected_holdout/outcome.json",
        "Input/processed/controlled_neural_world/six_context_outcome.json",
    ):
        with pytest.raises(ControlledWorldContractError, match="Forbidden"):
            reject_forbidden_path_tokens(path, label="forbidden")
