from __future__ import annotations

from collections.abc import Mapping
from typing import TypeVar

from Libs.migrations.legacy_naming import (
    CANONICAL_REPLACEMENT,
    DEPRECATION_VERSION,
    REMOVAL_GATE,
    TEST_COVERAGE,
    WARNING_BEHAVIOR,
    migrate_legacy_text,
    warn_legacy_use,
)


TensorValue = TypeVar("TensorValue")


def migrate_legacy_checkpoint_keys(
    state_dict: Mapping[str, TensorValue],
) -> dict[str, TensorValue]:
    """Map deprecated checkpoint keys without changing tensor values or scientific state."""
    migrated_keys = {raw_key: migrate_legacy_text(str(raw_key)) for raw_key in state_dict}
    if all(str(raw_key) == canonical_key for raw_key, canonical_key in migrated_keys.items()):
        return dict(state_dict)
    warn_legacy_use("checkpoint state dictionary", stacklevel=3)
    migrated: dict[str, TensorValue] = {}
    for raw_key, value in state_dict.items():
        canonical_key = migrated_keys[raw_key]
        if canonical_key in migrated:
            raise ValueError(f"Duplicate canonical and legacy checkpoint registration: {canonical_key}")
        migrated[canonical_key] = value
    return migrated


def migrate_mind_dino_checkpoint_keys(
    state_dict: Mapping[str, TensorValue],
) -> dict[str, TensorValue]:
    return migrate_legacy_checkpoint_keys(state_dict)


__all__ = [
    "CANONICAL_REPLACEMENT",
    "DEPRECATION_VERSION",
    "REMOVAL_GATE",
    "TEST_COVERAGE",
    "WARNING_BEHAVIOR",
    "migrate_legacy_checkpoint_keys",
    "migrate_mind_dino_checkpoint_keys",
]
