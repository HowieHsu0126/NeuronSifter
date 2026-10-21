from pathlib import Path

import torch

from Libs.models.modules.world_action_encoder import WorldActionEncoder
from Libs.utils.data.controlled_world_contracts import FAMILY_UIDS
from Libs.utils.data.controlled_world_tensorization import ControlledWorldTensorDataset, tensorize_manifest
from Libs.utils.exp_utils import load_config_stack


ROOT = Path(__file__).resolve().parents[2]


def test_real_training_manifest_tensorizes_three_families_with_causal_split() -> None:
    config = load_config_stack(ROOT / "Libs/configs/config_stack.yaml")
    manifest_path = config["operational_twin"]["manifests"]["training"]
    payload = tensorize_manifest(
        ROOT / manifest_path,
        "training",
        config,
        ROOT,
    )
    assert {row["family_uid"] for row in payload["records"]} == set(FAMILY_UIDS)
    assert len(payload["records"]) == 72
    for row in payload["records"]:
        assert row["input_observation"].shape == row["target_observation"].shape
        assert not torch.any(row["input_mask"] & row["target_mask"])
        assert row["input_mask"].any() and row["target_mask"].any()
        assert torch.count_nonzero(row["input_observation"][~row["input_mask"]]) == 0
        assert row["counterfactual_pair_uid"]
        assert row["latent_world_uid"]
        assert row["action_program_uid"]
    sample = ControlledWorldTensorDataset(payload, WorldActionEncoder())[0]
    assert sample["missingness_mask"].dtype == torch.bool
    assert not torch.equal(sample["action_encoding"], sample["no_action_encoding"])
