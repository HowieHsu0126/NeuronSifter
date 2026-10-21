from __future__ import annotations

from typing import Any

from Libs.migrations.legacy_naming import (
    CANONICAL_REPLACEMENT,
    DEPRECATION_VERSION,
    REMOVAL_GATE,
    TEST_COVERAGE,
    WARNING_BEHAVIOR,
    warn_legacy_use,
)
from Libs.models.modules.neuronsifter_world_action import (
    NeuronSifterWorldActionBlock,
    NeuronSifterWorldActionOperator,
)
from Libs.utils.model.neuronsifter_scale_registry import apply_neuronsifter_scale


class MindDinoMicroenvironmentOperator(NeuronSifterWorldActionOperator):
    """Deprecated construction alias; runtime output remains canonical."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warn_legacy_use("operator class", stacklevel=2)
        super().__init__(*args, **kwargs)


class MindDinoBlock(NeuronSifterWorldActionBlock):
    """Deprecated construction alias for serialized client imports."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warn_legacy_use("operator block class", stacklevel=2)
        super().__init__(*args, **kwargs)


def apply_mind_dino_scale(*args: Any, **kwargs: Any) -> dict[str, Any]:
    warn_legacy_use("scale application function", stacklevel=2)
    return apply_neuronsifter_scale(*args, **kwargs)


__all__ = [
    "CANONICAL_REPLACEMENT",
    "DEPRECATION_VERSION",
    "MindDinoBlock",
    "MindDinoMicroenvironmentOperator",
    "REMOVAL_GATE",
    "TEST_COVERAGE",
    "WARNING_BEHAVIOR",
    "apply_mind_dino_scale",
]
