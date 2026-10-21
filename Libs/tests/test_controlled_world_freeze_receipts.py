from __future__ import annotations

from pathlib import Path

from Libs.utils.data.controlled_world_benchmark_manifests import PARTITIONS, validate_benchmark_config
from Libs.utils.data.raw_data_registry import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"


def test_freeze_receipt_policy_keeps_access_receipts_unopened_without_materialization() -> None:
    config = load_config(CONFIG_PATH)
    bench = validate_benchmark_config(config)
    assert bench["freeze_policy"]["result_access_before_freeze"] == 0
    assert bench["freeze_policy"]["test_outcome_access"] == 0
    assert bench["split_policy"]["test_partitions"] == list(PARTITIONS)
    assert bench["access_policy"] == {
        "six_context_outcome_access": 0,
        "protected_holdout_access": 0,
        "test_outcome_access": 0,
        "protected_result_access": 0,
    }
