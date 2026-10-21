from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config, repo_root
from Libs.utils.evaluation.regimen_acquisition import build_acquisition_frame, select_acquisition_batch


def test_acquisition_batch_respects_multiobjective_scoring() -> None:
    config = load_config(str(repo_root() / "Libs/configs/config_stack.yaml"))
    regimen_frame = pd.DataFrame(
        [
            {
                "regimen_uid": "regimen_a",
                "target_symbol_set": "APP",
                "route": "intrathecal",
                "schedule": "weekly",
                "pk_feasibility_score": "0.9",
            },
            {
                "regimen_uid": "regimen_b",
                "target_symbol_set": "APP",
                "route": "intravenous",
                "schedule": "daily",
                "pk_feasibility_score": "0.8",
            },
            {
                "regimen_uid": "regimen_c",
                "target_symbol_set": "GLP1R",
                "route": "oral",
                "schedule": "daily",
                "pk_feasibility_score": "0.7",
            },
        ]
    )
    acquisition_frame = build_acquisition_frame(
        regimen_frame,
        predicted_scores=[0.9, 0.88, 0.7],
        epistemic_uncertainty=[0.2, 0.6, 0.9],
        config=config,
    )
    selected = select_acquisition_batch(acquisition_frame, batch_size=2)
    assert selected.shape[0] == 2
    assert "acquisition_score" in selected.columns
    assert selected["route"].nunique() == 2


def test_acquisition_frame_supports_intervention_program_style_target_labels_and_validation_readiness() -> None:
    config = load_config(str(repo_root() / "Libs/configs/config_stack.yaml"))
    intervention_frame = pd.DataFrame(
        [
            {
                "intervention_program_uid": "intervention_a",
                "target_program_label_set": "APP",
                "route": "intrathecal",
                "schedule": "weekly",
                "pk_feasibility_score": "0.9",
                "validation_readiness_score": "0.8",
            },
            {
                "intervention_program_uid": "intervention_b",
                "target_program_label_set": "APP",
                "route": "oral",
                "schedule": "daily",
                "pk_feasibility_score": "0.5",
                "validation_readiness_score": "0.1",
            },
        ]
    )
    acquisition_frame = build_acquisition_frame(
        intervention_frame,
        predicted_scores=[0.8, 0.7],
        epistemic_uncertainty=[0.2, 0.9],
        config=config,
    )
    assert "validation_readiness" in acquisition_frame.columns
    assert acquisition_frame["acquisition_score"].max() >= acquisition_frame["acquisition_score"].min()
