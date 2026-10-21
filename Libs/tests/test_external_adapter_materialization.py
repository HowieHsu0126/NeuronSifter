from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.validators.validate_external_adapter_registry import validate_drugcomb_processed_catalog
from Libs.utils.data.raw_data_registry import load_config, read_tsv
from Libs.utils.evaluation.external_adapter_materialization import (
    build_drugcomb_processed_catalogs,
    dgidb_batches,
    parse_dgidb_gene_nodes,
)


def test_dgidb_api_response_parser_keeps_target_universe_interactions() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    dgidb_config = config["validation"]["external_adapter_registry"]["materialization"]["dgidb"]
    nodes = [
        {
            "name": "APP",
            "conceptId": "hgnc:620",
            "interactions": [
                {
                    "id": "interaction-app-tramiprosate",
                    "interactionScore": 0.72,
                    "interactionTypes": [{"type": "inhibitor", "directionality": "decreases"}],
                    "publications": [{"citation": "fixture citation", "pmid": "12345"}],
                    "sources": [{"id": "fixture", "fullName": "Fixture Source"}],
                    "drug": {"name": "Tramiprosate", "conceptId": "chembl:fixture"},
                    "gene": {"name": "APP", "conceptId": "hgnc:620"},
                }
            ],
        }
    ]
    rows = parse_dgidb_gene_nodes(
        nodes,
        {"APP": "target_app"},
        {"TRAMIPROSATE": "compound_tramiprosate"},
        dgidb_config,
        "2026-04-26",
    )
    assert dgidb_batches(["APP", "BACE1", "MAPT"], 2) == [["APP", "BACE1"], ["MAPT"]]
    assert len(rows) == 1
    assert rows[0]["drug_in_neuronsifter_universe"] == "yes"
    assert rows[0]["matched_target_uid"] == "target_app"
    assert rows[0]["matched_compound_uid"] == "compound_tramiprosate"
    assert rows[0]["provenance_level"] == "dgidb_api_target_universe_query"


def write_drugcomb_fixture_files(tmp_path: Path) -> tuple[Path, Path]:
    dose_path = tmp_path / "drugcomb_data_v1_4.csv"
    dose_path.write_text(
        "\n".join(
            [
                "block_id,conc_r,conc_c,inhibition,drug_row,drug_col,conc_r_unit,conc_c_unit,cell_line_name,drug_row_cid,drug_col_cid,cellosaurus_accession,study_name",
                "1,0.1,0.2,12.5,DrugA,DrugB,uM,uM,CellA,111,222,CVCL_0001,StudyA",
                "1,0.1,1.0,22.5,DrugA,DrugB,uM,uM,CellA,111,222,CVCL_0001,StudyA",
                "1,1.0,0.2,32.5,DrugA,DrugB,uM,uM,CellA,111,222,CVCL_0001,StudyA",
                "2,0.1,0.2,8.0,DrugC,DrugD,uM,uM,CellB,333,444,CVCL_0002,StudyB",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path = tmp_path / "summary_v_1_5.csv"
    summary_path.write_text(
        "\n".join(
            [
                "block_id,drug_row,drug_col,cell_line_name,study_name,tissue_name,conc_row_unit,conc_col_unit,ic50_row,ic50_col,ri_row,ri_col,css_row,css_col,css_ri,S_sum,S_mean,S_max,synergy_zip,synergy_loewe,synergy_hsa,synergy_bliss,drug_row_clinical_phase,drug_col_clinical_phase,drug_row_target_name,drug_col_target_name",
                "1,DrugA,DrugB,CellA,StudyA,brain,uM,uM,1.0,2.0,0.2,0.3,40.0,35.0,37.5,12.0,4.0,7.0,5.5,3.2,1.1,2.4,approved,phase_2,TargetA,TargetB",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return dose_path, summary_path


def drugcomb_temp_config(tmp_path: Path) -> dict:
    config = deepcopy(load_config(REPO_ROOT / "Libs/configs/config_stack.yaml"))
    registry = config["validation"]["external_adapter_registry"]
    registry["local_root"] = tmp_path.as_posix()
    registry["relative_paths"]["drugcomb_combination_prior_catalog"] = "drugcomb_combination_prior_catalog.tsv"
    registry["relative_paths"]["drugcomb_dose_response_matrix_dir"] = "drugcomb_dose_response_matrix_catalog"
    registry["relative_paths"]["drugcomb_dose_response_manifest"] = "drugcomb_dose_response_manifest.json"
    dose_path, summary_path = write_drugcomb_fixture_files(tmp_path)
    materialization = registry["materialization"]["drugcomb"]
    materialization["dose_response_csv_path"] = dose_path.as_posix()
    materialization["synergy_summary_csv_path"] = summary_path.as_posix()
    materialization["chunk_row_count"] = 2
    return config


def test_drugcomb_full_matrix_builder_links_summary_and_preserves_hashes(tmp_path: Path) -> None:
    config = drugcomb_temp_config(tmp_path)
    prior_path, matrix_dir = build_drugcomb_processed_catalogs(config)
    validate_drugcomb_processed_catalog(prior_path, config)
    rows = read_tsv(prior_path)
    manifest = json.loads((tmp_path / "drugcomb_dose_response_manifest.json").read_text(encoding="utf-8"))
    part_files = sorted(matrix_dir.glob("part-*.tsv"))
    first_part_rows = read_tsv(part_files[0])
    assert len(rows) == 1
    assert rows[0]["dose_response_row_count"] == "3"
    assert rows[0]["dose_response_part_path"].endswith("part-00000.tsv")
    assert len(first_part_rows) == 3
    assert manifest["full_matrix_flag"] == "yes"
    assert manifest["dose_response_source_sha256"]
    assert manifest["synergy_summary_source_sha256"]


def test_drugcomb_strict_validator_rejects_smoke_matrix_manifest(tmp_path: Path) -> None:
    config = drugcomb_temp_config(tmp_path)
    prior_path, _ = build_drugcomb_processed_catalogs(config, max_dose_response_rows=3)
    with pytest.raises(ValueError, match="full matrix manifest"):
        validate_drugcomb_processed_catalog(prior_path, config)
