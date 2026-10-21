from __future__ import annotations

from Libs.utils.data.raw_data_registry import load_config
from Libs.utils.model.dynamics_fields import dynamics_process_context


def test_dynamics_process_context_uses_configured_start_method() -> None:
    config = load_config("Libs/configs/config_stack.yaml")
    context = dynamics_process_context(config)
    assert context.get_start_method() == config["dynamics_ready_registry"]["process_start_method"]
    assert context.get_start_method() == "spawn"
