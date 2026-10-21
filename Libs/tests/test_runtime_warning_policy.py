from __future__ import annotations

import warnings

import pytest

from Libs.utils.data.raw_data_registry import apply_warning_policy, load_config


def test_loaded_config_applies_runtime_warning_policy() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        config = load_config("Libs/configs/config_stack.yaml")
        assert config["runtime_warning_policy"]["action"] == "error"
        with pytest.raises(UserWarning):
            warnings.warn("runtime warning policy contract", UserWarning)


def test_warning_policy_rejects_non_warning_category() -> None:
    with pytest.raises(ValueError):
        apply_warning_policy({"runtime_warning_policy": {"enabled": True, "action": "error", "category": "Exception"}})
