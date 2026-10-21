from __future__ import annotations

import json
from pathlib import Path
import subprocess

from jsonschema import Draft202012Validator
import pytest
import yaml

from Libs.utils.data.controlled_world_benchmark_manifests import OOD_CLASSES, PARTITIONS, build_ood_manifest, validate_benchmark_config
from Libs.utils.data.controlled_world_contracts import ControlledWorldContractError
from Libs.utils.data.raw_data_registry import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"
SCHEMA_PATH = REPO_ROOT / "Libs/configs/schema/controlled_world_benchmark_manifest.schema.yaml"


def test_benchmark_config_fail_closed_and_declares_lineage_and_retention() -> None:
    config = load_config(CONFIG_PATH)
    bench = validate_benchmark_config(config)
    assert bench["latent_lineage_policy"]["require_latent_generator_lineage_separation_by_role"] is True
    assert bench["statistical_policy"]["bootstrap_method"] == "family_stratified_paired_bootstrap"
    assert bench["statistical_policy"]["failure_and_undefined_handling"]["retain_every_case"] is True
    assert bench["access_policy"] == {"six_context_outcome_access": 0, "protected_holdout_access": 0, "test_outcome_access": 0, "protected_result_access": 0}
    bad = json.loads(json.dumps(config))
    bad["controlled_world_benchmark_qualification"]["source_firewall"]["prohibited_source_routes"].append("new_route")
    with pytest.raises(ControlledWorldContractError):
        validate_benchmark_config(bad)


def test_benchmark_config_records_sealed_partitions_without_materializing_test_manifest() -> None:
    config = load_config(CONFIG_PATH)
    bench = validate_benchmark_config(config)
    schema = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    assert bench["split_policy"]["test_partitions"] == list(PARTITIONS)
    assert bench["freeze_policy"]["freeze_before_any_test_result_access"] is True


def test_ood_manifest_covers_all_classes_and_retains_failures() -> None:
    config = load_config(CONFIG_PATH)
    manifests = {
        role: {
            "worlds": [
                {
                    "world_uid": f"{role}_{family_uid}",
                    "seed_identity": f"{role}_{family_uid}_seed",
                    "latent_generator_lineage_uid": f"{role}_{family_uid}_lineage",
                }
                for family_uid in (
                    "electrophysiology_conductance_world",
                    "calcium_synaptic_world",
                    "molecular_pharmacological_causal_world",
                )
            ]
        }
        for role in ("training", "development", "calibration")
    }
    ood = build_ood_manifest(config, manifests, repository_root=REPO_ROOT)
    assert tuple(ood["ood_classes"]) == OOD_CLASSES
    assert {row["ood_class"] for row in ood["records"]} == set(OOD_CLASSES)
    assert any(row["failure_record"] is not None for row in ood["records"])
    assert ood["access_counters"] == {"six_context_outcome_access": 0, "protected_holdout_access": 0, "test_outcome_access": 0}


def test_historical_benchmark_blob_identities_are_preserved_without_opening_test_manifest() -> None:
    expected = {
        "Input/processed/controlled_neural_world/benchmark/training_manifest.json": "7cf6c88c22c3af02b599611feaa7815070c61b64",
        "Input/processed/controlled_neural_world/benchmark/development_manifest.json": "e0d3544b9352b6cb5113936421961f478d90568a",
        "Input/processed/controlled_neural_world/benchmark/calibration_manifest.json": "c06080ae7241bb04b41df7a475889d33b28fbd8d",
        "Input/processed/controlled_neural_world/benchmark/test_manifest.json": "674c793eb70bc25bf4231fae15e7a8c932272436",
    }
    for path, blob_id in expected.items():
        observed = subprocess.check_output(
            ["git", "rev-parse", f"HEAD:{path}"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
        assert observed == blob_id
