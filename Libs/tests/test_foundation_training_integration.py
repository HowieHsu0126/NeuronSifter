from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.model.screening_core import (
    counterfactual_pairwise_loss,
    simulation_teacher_factor_loss,
    world_state_factorization_loss,
    world_state_target_loss,
)
from Libs.utils.evaluation.screening_datasets import (
    counterfactual_pair_indices,
    simulation_teacher_factor_masks,
    world_state_support_vectors,
    world_state_supervision_masks,
)


def test_world_state_support_vectors_extract_hidden_state_and_missing_modalities() -> None:
    metadata = pd.DataFrame([{"sample_uid": "sample_a"}, {"sample_uid": "sample_b"}])
    world_state = pd.DataFrame(
        [
            {
                "sample_uid": "sample_a",
                "hidden_state_support_flag": "yes",
                "missing_modality_set": "morphology|electrophysiology",
            },
            {
                "sample_uid": "sample_b",
                "hidden_state_support_flag": "no",
                "missing_modality_set": "",
            },
        ]
    )
    payload = world_state_support_vectors(metadata, world_state)
    assert payload["hidden_state_support"].tolist() == [1.0, 0.0]
    assert payload["missing_modality_count"].tolist() == [2.0, 0.0]


def test_world_state_supervision_masks_follow_direct_factor_support() -> None:
    metadata = pd.DataFrame([{"sample_uid": "sample_a"}, {"sample_uid": "sample_b"}])
    world_state = pd.DataFrame(
        [
            {
                "sample_uid": "sample_a",
                "factor_supervision_mask_payload": '{"z_tx": 1.0, "z_pk": 0.0, "z_env": 1.0, "z_ephys": 0.0, "z_morph": 0.0}',
            },
            {
                "sample_uid": "sample_b",
                "factor_supervision_mask_payload": '{"z_tx": 0.0, "z_pk": 1.0, "z_env": 1.0, "z_ephys": 1.0, "z_morph": 0.0}',
            },
        ]
    )
    payload = world_state_supervision_masks(metadata, world_state)
    assert payload["z_tx"].tolist() == [1.0, 0.0]
    assert payload["z_pk"].tolist() == [0.0, 1.0]
    assert payload["z_env"].tolist() == [1.0, 1.0]


def test_simulation_teacher_factor_masks_follow_regimen_uid_scope() -> None:
    regimen_frame = pd.DataFrame([{"regimen_uid": "regimen_a"}, {"regimen_uid": "regimen_b"}])
    teacher_frame = pd.DataFrame(
        [
            {"disease_program": "ad_control", "regimen_uid": "regimen_a", "factor_scope": "z_pk|z_env"},
            {"disease_program": "ad_control", "regimen_uid": "regimen_b", "factor_scope": "z_ephys"},
        ]
    )
    masks = simulation_teacher_factor_masks(regimen_frame, teacher_frame, "ad_control")
    assert masks["z_pk"].tolist() == [1.0, 0.0]
    assert masks["z_env"].tolist() == [1.0, 0.0]
    assert masks["z_ephys"].tolist() == [0.0, 1.0]


def test_counterfactual_pair_indices_map_selected_regimens() -> None:
    regimen_frame = pd.DataFrame([{"regimen_uid": "regimen_a"}, {"regimen_uid": "regimen_b"}, {"regimen_uid": "regimen_c"}])
    counterfactual_frame = pd.DataFrame(
        [
            {"disease_program": "ad_control", "baseline_regimen_uid": "regimen_a", "counterfactual_regimen_uid": "regimen_b"},
            {"disease_program": "ad_control", "baseline_regimen_uid": "regimen_b", "counterfactual_regimen_uid": "regimen_c"},
        ]
    )
    pairs = counterfactual_pair_indices(regimen_frame, counterfactual_frame, "ad_control")
    assert pairs == [(0, 1), (1, 2)]


def test_world_state_factorization_loss_returns_positive_scalar() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    world_state_payload = {
        "z_tx": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        "z_pk": torch.tensor([[0.5, 0.1], [0.4, 0.2]], dtype=torch.float32),
        "z_env": torch.tensor([[0.2, 0.3], [0.3, 0.4]], dtype=torch.float32),
        "z_ephys": torch.tensor([[0.1], [0.2]], dtype=torch.float32),
        "z_morph": torch.tensor([[0.9, 0.8], [0.7, 0.6]], dtype=torch.float32),
    }
    loss = world_state_factorization_loss(
        world_state_payload,
        torch.tensor([1.0, 1.0], dtype=torch.float32),
        torch.tensor([2.0, 2.0], dtype=torch.float32),
        config,
    )
    assert float(loss.item()) >= 0.0


def test_world_state_target_loss_respects_factor_masks() -> None:
    world_state_payload = {
        "z_tx": torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        "z_pk": torch.tensor([[2.0, 2.0]], dtype=torch.float32),
        "z_env": torch.tensor([[3.0, 3.0]], dtype=torch.float32),
        "z_ephys": torch.tensor([[4.0]], dtype=torch.float32),
        "z_morph": torch.tensor([[5.0, 5.0]], dtype=torch.float32),
    }
    factor_targets = {
        "z_tx": torch.tensor([0.0], dtype=torch.float32),
        "z_pk": torch.tensor([0.0], dtype=torch.float32),
        "z_env": torch.tensor([0.0], dtype=torch.float32),
        "z_ephys": torch.tensor([0.0], dtype=torch.float32),
        "z_morph": torch.tensor([0.0], dtype=torch.float32),
    }
    factor_masks = {
        "z_tx": torch.tensor([1.0], dtype=torch.float32),
        "z_pk": torch.tensor([0.0], dtype=torch.float32),
        "z_env": torch.tensor([0.0], dtype=torch.float32),
        "z_ephys": torch.tensor([0.0], dtype=torch.float32),
        "z_morph": torch.tensor([0.0], dtype=torch.float32),
    }
    loss = world_state_target_loss(world_state_payload, factor_targets, factor_masks)
    assert float(loss.item()) == 1.0


def test_simulation_teacher_factor_loss_uses_factor_masks() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    predicted = torch.zeros((2, 2, 4, 3, 2, 7), dtype=torch.float32)
    target = torch.ones((2, 2, 4, 3, 2, 7), dtype=torch.float32)
    factor_masks = {
        "z_pk": torch.tensor([1.0, 0.0], dtype=torch.float32),
        "z_env": torch.tensor([0.0, 1.0], dtype=torch.float32),
        "z_ephys": torch.tensor([1.0, 1.0], dtype=torch.float32),
    }
    loss = simulation_teacher_factor_loss(predicted, target, factor_masks, config)
    assert float(loss.item()) > 0.0


def test_counterfactual_pairwise_loss_penalizes_wrong_ordering() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    predicted_scores = torch.tensor([[0.1, 0.0]], dtype=torch.float32)
    target_scores = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
    loss = counterfactual_pairwise_loss(predicted_scores, target_scores, [(0, 1)], config)
    assert float(loss.item()) > 0.0
