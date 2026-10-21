from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.compat.legacy_mind_dino import (
    CANONICAL_REPLACEMENT,
    DEPRECATION_VERSION,
    MindDinoMicroenvironmentOperator,
    REMOVAL_GATE,
    TEST_COVERAGE,
    WARNING_BEHAVIOR,
)
from Libs.migrations.legacy_checkpoint import migrate_mind_dino_checkpoint_keys
from Libs.migrations.legacy_config import migrate_mind_dino_config
from Libs.models.modules.neuronsifter_world_action import NeuronSifterWorldActionOperator
from Libs.models.modules.mind_dino_operator import MindDinoMicroenvironmentOperator as LegacyImportOperator
from Libs.scripts.reports.build_paper_results import scale_registry_audit
from Libs.scripts.services.serve_screening_api import scale_aware_response
from Libs.utils.data.g0_naming import FORBIDDEN_CATEGORIES, scan_naming_residuals
from Libs.utils.data.raw_data_registry import load_config


CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"
CANONICAL_MODEL_NAMES = {
    "NeuronSifter-Debug",
    "NeuronSifter-Student",
    "NeuronSifter-Large",
    "NeuronSifter-XL",
    "NeuronSifter-Foundation",
}


def test_import_compatibility_is_deprecated_and_resolves_to_canonical_class() -> None:
    assert issubclass(MindDinoMicroenvironmentOperator, NeuronSifterWorldActionOperator)
    assert LegacyImportOperator is MindDinoMicroenvironmentOperator
    assert DEPRECATION_VERSION == "2.0.0"
    assert REMOVAL_GATE
    assert CANONICAL_REPLACEMENT == "NeuronSifter canonical identifiers"
    assert "DeprecationWarning" in WARNING_BEHAVIOR
    assert TEST_COVERAGE.endswith("test_g0_naming_migration.py")


def test_config_key_migration_warns_and_serializes_only_canonical_names() -> None:
    legacy = {
        "mind_dino_scale_registry": {
            "default_training_checkpoint_scale": "mind_dino_large",
            "operator_kind": "mind_dino_microenvironment_operator",
        }
    }
    with pytest.warns(DeprecationWarning):
        migrated = migrate_mind_dino_config(legacy)
    assert migrated == {
        "neuronsifter_scale_registry": {
            "default_training_checkpoint_scale": "neuronsifter_large",
            "operator_kind": "neuronsifter_world_action",
        }
    }
    assert "mind_dino" not in json.dumps(migrated, sort_keys=True)


def test_checkpoint_key_mapping_preserves_values_and_rejects_duplicate_registration() -> None:
    marker = object()
    with pytest.warns(DeprecationWarning):
        migrated = migrate_mind_dino_checkpoint_keys({"mind_dino_operator.blocks.0.weight": marker})
    assert migrated == {"neuronsifter_world_action_operator.blocks.0.weight": marker}

    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="Duplicate canonical and legacy"):
        migrate_mind_dino_checkpoint_keys(
            {
                "mind_dino_operator.blocks.0.weight": marker,
                "neuronsifter_world_action_operator.blocks.0.weight": marker,
            }
        )


def test_config_migration_rejects_parallel_legacy_and_canonical_registration() -> None:
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="Duplicate canonical and legacy"):
        migrate_mind_dino_config(
            {
                "mind_dino_scale_registry": {},
                "neuronsifter_scale_registry": {},
            }
        )


def test_api_model_name_report_labels_and_slurm_suffix_are_canonical() -> None:
    config = load_config(CONFIG_PATH)
    public_identity = config["iclr_design_contract_union"]["public_identity"]
    assert set(public_identity["canonical_component_names"]) == {
        "NeuronSifterFoundationStateModel",
        "NeuronSifterWorldActionOperator",
        "NeuronSifterToolAcquisitionPolicy",
        "NeuronSifterReliabilityGate",
    }
    response = scale_aware_response(
        {"model_card": {"model_version": "test", "checkpoint_id": "test:large"}},
        config,
        {},
        {"candidate_count": 1},
    )
    assert response["model_name"] == "NeuronSifter-Large"
    report_labels = {row["scale_label"] for row in scale_registry_audit(config)["checkpoint_scales"]}
    assert report_labels == CANONICAL_MODEL_NAMES
    assert config["neuronsifter_scale_registry"]["fsdp_resource_profiles"]["xl_four_gpu_smoke"]["job_suffix"] == "neuronsifter_xl_fsdp_smoke"


def test_forbidden_residual_and_public_alias_propagation_are_detected(tmp_path: Path) -> None:
    active_file = tmp_path / "Libs/active.py"
    active_file.parent.mkdir()
    active_file.write_text('MODEL_KIND = "mind_dino_microenvironment_operator"\n', encoding="utf-8")
    public_file = tmp_path / "README.md"
    public_file.write_text("MIND-DiNO is the current model.\n", encoding="utf-8")
    config = load_config(CONFIG_PATH)
    policy = config["iclr_design_contract_union"]["g0_naming_audit"]
    policy["scan_paths"] = ["Libs", "README.md"]
    policy["excluded_path_prefixes"] = []
    policy["allowed_path_categories"] = []
    rows = scan_naming_residuals(config, tmp_path)
    assert {row["category"] for row in rows} == {"forbidden_active_code", "forbidden_public_text"}
    assert len(rows) == 2


def test_repository_residuals_are_confined_to_approved_contexts() -> None:
    rows = scan_naming_residuals(load_config(CONFIG_PATH))
    assert not [row for row in rows if row["category"] in FORBIDDEN_CATEGORIES]
    assert rows
