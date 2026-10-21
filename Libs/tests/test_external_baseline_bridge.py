from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.external_baseline_bridge import baseline_torch_device, dense_matrix_to_ragged_batch


def test_dense_matrix_to_ragged_batch_keeps_nonzero_features() -> None:
    matrix = np.asarray(
        [
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    indices, values, offsets = dense_matrix_to_ragged_batch(matrix)
    assert indices.tolist() == [2, 1]
    assert values.tolist() == [2.0, 0.0]
    assert offsets.tolist() == [0, 1, 2]


def test_baseline_torch_device_respects_cpu_only_slurm_allocation(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.delenv("SLURM_STEP_GPUS", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    assert baseline_torch_device().type == "cpu"
