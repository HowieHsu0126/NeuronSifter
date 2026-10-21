from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from Libs.migrations.legacy_naming import (
    CANONICAL_REPLACEMENT,
    DEPRECATION_VERSION,
    REMOVAL_GATE,
    TEST_COVERAGE,
    WARNING_BEHAVIOR,
    migrate_legacy_text,
    warn_legacy_use,
)


def migrate_mind_dino_config(payload: object) -> object:
    """Return a canonical deep copy and reject mixed-name key collisions."""
    warn_legacy_use("serialized config", stacklevel=2)

    def migrate(value: object) -> object:
        if isinstance(value, Mapping):
            result: dict[str, object] = {}
            for raw_key, raw_value in value.items():
                canonical_key = migrate_legacy_text(str(raw_key))
                if canonical_key in result:
                    raise ValueError(f"Duplicate canonical and legacy config registration: {canonical_key}")
                result[canonical_key] = migrate(raw_value)
            return result
        if isinstance(value, list):
            return [migrate(item) for item in value]
        if isinstance(value, tuple):
            return tuple(migrate(item) for item in value)
        if isinstance(value, str):
            return migrate_legacy_text(value)
        return value

    return migrate(payload)


__all__ = [
    "CANONICAL_REPLACEMENT",
    "DEPRECATION_VERSION",
    "REMOVAL_GATE",
    "TEST_COVERAGE",
    "WARNING_BEHAVIOR",
    "migrate_mind_dino_config",
]
