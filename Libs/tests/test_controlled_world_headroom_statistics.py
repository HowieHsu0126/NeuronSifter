from __future__ import annotations

from pathlib import Path

from Libs.utils.data.controlled_world_contracts import ControlledWorldContractError
from Libs.utils.data.controlled_world_headroom_statistics import _ensure_no_oracle_fields
from Libs.utils.data.raw_data_registry import load_config
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"


def test_development_only_headroom_uses_world_as_unit_and_no_oracle_fields() -> None:
    config = load_config(CONFIG_PATH)
    policy = config["controlled_world_benchmark_qualification"]["headroom_policy"]
    assert policy["development_only"] is True
    assert policy["independent_unit"] == "world"
    assert policy["public_record_excludes_hidden_oracle_fields"] is True


def test_qualification_terminal_is_exact_and_family_spanning_gate_is_data_driven() -> None:
    config = load_config(CONFIG_PATH)
    terminals = config["controlled_world_benchmark_qualification"]["terminal_policy"]["allowed_terminals"]
    assert set(terminals) == {
        "CONTROLLED_NEURAL_WORLD_BENCHMARK_QUALIFIED",
        "CONTROLLED_NEURAL_WORLD_BENCHMARK_NOT_QUALIFIED",
    }
    assert config["controlled_world_benchmark_qualification"]["statistical_policy"]["bootstrap_method"] == "family_stratified_paired_bootstrap"


def test_oracle_firewall_allows_only_required_zero_access_counter_keys() -> None:
    _ensure_no_oracle_fields({"access_counters": {"six_context_outcome_access": 0, "protected_holdout_access": 0, "test_outcome_access": 0}})
    with pytest.raises(ControlledWorldContractError, match="protected_access_nonzero"):
        _ensure_no_oracle_fields({"access_counters": {"six_context_outcome_access": 1}})
    with pytest.raises(ControlledWorldContractError, match="oracle_access_forbidden"):
        _ensure_no_oracle_fields({"payload": {"oracle_parameters": {"leak": 1.0}}})
    with pytest.raises(ControlledWorldContractError, match="oracle_access_forbidden"):
        _ensure_no_oracle_fields({"payload": "contains hidden_optimum marker"})
