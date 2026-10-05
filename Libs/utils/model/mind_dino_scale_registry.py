"""Deprecated import shim for canonical NeuronSifter scale utilities."""

from Libs.compat.legacy_mind_dino import apply_mind_dino_scale
from Libs.migrations.legacy_naming import (
    CANONICAL_REPLACEMENT,
    DEPRECATION_VERSION,
    REMOVAL_GATE,
    TEST_COVERAGE,
    WARNING_BEHAVIOR,
    warn_legacy_use,
)
from Libs.utils.model.neuronsifter_scale_registry import (
    canonical_model_name,
    checkpoint_scale_config,
    default_checkpoint_scale,
    default_inference_profile,
    fsdp_profile_config,
    inference_profile_config,
    registry,
    scale_parameter_estimate,
    scale_route_metadata,
)


def resolve_legacy_scale_registry() -> object:
    warn_legacy_use("deprecated scale-registry import module", stacklevel=2)
    return registry


__all__ = [
    "CANONICAL_REPLACEMENT",
    "DEPRECATION_VERSION",
    "REMOVAL_GATE",
    "TEST_COVERAGE",
    "WARNING_BEHAVIOR",
    "apply_mind_dino_scale",
    "canonical_model_name",
    "checkpoint_scale_config",
    "default_checkpoint_scale",
    "default_inference_profile",
    "fsdp_profile_config",
    "inference_profile_config",
    "registry",
    "resolve_legacy_scale_registry",
    "scale_parameter_estimate",
    "scale_route_metadata",
]
