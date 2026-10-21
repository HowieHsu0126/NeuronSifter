from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_external_adapter_registry import adapter_rows
from Libs.scripts.validators.validate_external_adapter_registry import validate_external_adapter_rows
from Libs.utils.data.raw_data_registry import load_config


def test_external_adapter_registry_declares_all_roles_and_provenance() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    rows = adapter_rows(config)
    required_sources = {
        "open_targets",
        "chembl",
        "bindingdb",
        "drugcentral",
        "dgidb",
        "drugcomb",
        "lincs_clue_sigcom",
        "osp_pksim_mobi",
        "rdkit",
        "moleculenet_deepchem",
        "autodock_vina_gnina",
    }
    rows_by_source = {row["source_key"]: row for row in rows}
    assert set(rows_by_source) == required_sources
    for source_key, row in rows_by_source.items():
        assert row["adapter_role"]
        assert row["provenance_level"]
        assert row["validation_policy"]
        assert row["method_contribution_flag"] == "no", source_key
        assert row["public_url"].startswith("https://")
    assert rows_by_source["moleculenet_deepchem"]["required_processed_artifact_flag"] == "yes"
    assert rows_by_source["moleculenet_deepchem"]["validation_policy"] == "strict_processed_molecular_property_baseline_required"


def test_dgidb_and_drugcomb_strict_mode_rejects_raw_metadata_only() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    rows = adapter_rows(config)
    missing_catalog_rows = []
    for row in rows:
        updated_row = dict(row)
        if updated_row["source_key"] in {"dgidb", "drugcomb"}:
            updated_row["processed_artifact_path"] = "Input/processed/external_adapters/raw_metadata_only_missing_catalog.tsv"
        missing_catalog_rows.append(updated_row)
    rows_by_source = {row["source_key"]: row for row in missing_catalog_rows}
    assert rows_by_source["dgidb"]["validation_policy"] == "strict_processed_interactions_required"
    assert rows_by_source["drugcomb"]["validation_policy"] == "strict_processed_synergy_required"
    with pytest.raises(FileNotFoundError):
        validate_external_adapter_rows(missing_catalog_rows, config, strict_processed=True)
