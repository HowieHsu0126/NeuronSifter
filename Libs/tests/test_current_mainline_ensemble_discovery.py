from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.exp.evaluation.evaluate_screening_system import discovered_ensemble_checkpoint_paths


def write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def test_current_mainline_ensemble_discovery_filters_by_fold_and_contract(tmp_path: Path) -> None:
    artifact_root = Path("Output") / "pytest_current_mainline_discovery" / tmp_path.name
    benchmark_template = "dynamics_twin_ad_mainline_seed_{seed}"
    fold_name = "heldout_st000433"
    config = {
        "screening_pipeline": {
            "screening_benchmark": {
                "benchmark_name": benchmark_template.format(seed=7),
                "ensemble_uq": {
                    "member_seed_values": [7, 11, 19],
                    "minimum_member_count": 2,
                    "current_mainline_discovery": {
                        "artifact_root": str(artifact_root),
                        "benchmark_name_template": benchmark_template,
                        "checkpoint_relative_path": "folds/{fold_name}/ad_screening_system/model_weights.pt",
                        "manifest_relative_path": "folds/{fold_name}/ad_screening_system/training_manifest.json",
                        "allowed_fold_names": [fold_name],
                        "expected_model_name": "ad_screening_system",
                        "expected_model_contract_version": "screening_system_multiscale_v1",
                    },
                },
            }
        }
    }
    for seed_value in [7, 11]:
        checkpoint_path = REPO_ROOT / artifact_root / benchmark_template.format(seed=seed_value) / "folds" / fold_name / "ad_screening_system" / "model_weights.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(b"checkpoint")
        write_manifest(
            checkpoint_path.parent / "training_manifest.json",
            {
                "fold_name": fold_name,
                "benchmark_name": benchmark_template.format(seed=seed_value),
                "model_name": "ad_screening_system",
                "model_contract_version": "screening_system_multiscale_v1",
            },
        )
    rejected_checkpoint = REPO_ROOT / artifact_root / benchmark_template.format(seed=19) / "folds" / fold_name / "ad_screening_system" / "model_weights.pt"
    rejected_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    rejected_checkpoint.write_bytes(b"checkpoint")
    write_manifest(
        rejected_checkpoint.parent / "training_manifest.json",
        {
            "fold_name": "heldout_st004168",
            "benchmark_name": benchmark_template.format(seed=19),
            "model_name": "ad_screening_system",
            "model_contract_version": "screening_system_multiscale_v1",
        },
    )

    discovered_paths = discovered_ensemble_checkpoint_paths(
        config,
        fold_name,
        REPO_ROOT / artifact_root / benchmark_template.format(seed=7) / "folds" / fold_name / "ad_screening_system" / "model_weights.pt",
        "required",
    )
    assert len(discovered_paths) == 2
    assert all(path.exists() for path in discovered_paths)
    assert rejected_checkpoint not in discovered_paths
