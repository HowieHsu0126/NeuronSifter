"""Deprecated import shim for the canonical NeuronSifter world-action module."""

from Libs.compat.legacy_mind_dino import (
    CANONICAL_REPLACEMENT,
    DEPRECATION_VERSION,
    MindDinoBlock,
    MindDinoMicroenvironmentOperator,
    REMOVAL_GATE,
    TEST_COVERAGE,
    WARNING_BEHAVIOR,
)
from Libs.migrations.legacy_naming import warn_legacy_use
from Libs.models.modules.neuronsifter_world_action import GraphSpectralTemporalWaveletTokenizer


def resolve_legacy_operator_class() -> type[MindDinoMicroenvironmentOperator]:
    warn_legacy_use("deprecated operator import module", stacklevel=2)
    return MindDinoMicroenvironmentOperator


__all__ = [
    "CANONICAL_REPLACEMENT",
    "DEPRECATION_VERSION",
    "GraphSpectralTemporalWaveletTokenizer",
    "MindDinoBlock",
    "MindDinoMicroenvironmentOperator",
    "REMOVAL_GATE",
    "TEST_COVERAGE",
    "WARNING_BEHAVIOR",
    "resolve_legacy_operator_class",
]
