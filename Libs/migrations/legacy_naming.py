from __future__ import annotations

import re
import warnings


DEPRECATION_VERSION = "2.0.0"
REMOVAL_GATE = "g8_release_requires_zero_legacy_inputs"
CANONICAL_REPLACEMENT = "NeuronSifter canonical identifiers"
WARNING_BEHAVIOR = "emit DeprecationWarning for every compatibility or migration call"
TEST_COVERAGE = "Libs/tests/test_g0_naming_migration.py"

LEGACY_TO_CANONICAL_REPLACEMENTS = (
    ("Libs/models/modules/mind_dino_operator.py", "Libs/models/modules/neuronsifter_world_action.py"),
    ("Libs/utils/model/mind_dino_scale_registry.py", "Libs/utils/model/neuronsifter_scale_registry.py"),
    ("Libs/configs/model/mind_dino_scale_registry.yaml", "Libs/configs/model/neuronsifter_scale_registry.yaml"),
    ("submit_mind_dino_fsdp_smoke.sh", "submit_neuronsifter_fsdp_smoke.sh"),
    ("validate_mind_dino_stochastic_smoke.py", "validate_neuronsifter_stochastic_smoke.py"),
    ("mind_dino_microenvironment_active_parameters", "neuronsifter_world_action_active_parameters"),
    ("mind_dino_microenvironment_operator", "neuronsifter_world_action"),
    ("mind_dino_operator", "neuronsifter_world_action_operator"),
    ("mind_dino_scale_registry", "neuronsifter_scale_registry"),
    ("mind_dino_scale_selection", "neuronsifter_scale_selection"),
    ("mind_dino_foundation_candidate", "neuronsifter_foundation"),
    ("mind_dino_mainline", "neuronsifter_mainline"),
    ("mind_dino_scratch", "neuronsifter_scratch"),
    ("mind_dino_full", "neuronsifter_full"),
    ("deterministic_mind_dino", "deterministic_neuronsifter"),
    ("full_mind_dino_large", "full_neuronsifter_large"),
    ("stochastic_mind_dino_world_model", "stochastic_neuronsifter_world_action"),
    ("MindDinoMicroenvironmentOperator", "NeuronSifterWorldActionOperator"),
    ("MindDinoBlock", "NeuronSifterWorldActionBlock"),
    ("MIND-DiNO", "NeuronSifter"),
    ("MIND_DINO", "NEURONSIFTER"),
    ("mind_dino", "neuronsifter"),
)


def warn_legacy_use(context: str, stacklevel: int = 2) -> None:
    warnings.warn(
        f"Legacy NeuronSifter naming in {context} is deprecated since {DEPRECATION_VERSION}; "
        f"use {CANONICAL_REPLACEMENT}. Removal gate: {REMOVAL_GATE}.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def migrate_legacy_text(value: str) -> str:
    migrated = value
    for legacy, canonical in LEGACY_TO_CANONICAL_REPLACEMENTS:
        migrated = migrated.replace(legacy, canonical)
    migrated = re.sub(r"mind[-_ ]?dino", "neuronsifter", migrated, flags=re.IGNORECASE)
    return migrated


def canonical_replacement_for_legacy_symbol(matched_string: str, context: str) -> str:
    for legacy, canonical in LEGACY_TO_CANONICAL_REPLACEMENTS:
        if legacy in context or legacy == matched_string:
            return canonical
    return "NeuronSifter" if "-" in matched_string or " " in matched_string else "neuronsifter"
