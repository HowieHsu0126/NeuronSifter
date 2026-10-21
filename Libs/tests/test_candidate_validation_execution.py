from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.evaluation.candidate_validation_execution import (
    build_execution_rows,
    parse_gnina_score,
    resolve_required_binaries,
    write_candidate_validation_execution_catalog,
)
from Libs.utils.data.platform_contracts import candidate_validation_rows
from Libs.utils.data.raw_data_registry import load_config, write_tsv


def repo_relative_tmp(tmp_path: Path) -> Path:
    return Path("Output") / "pytest_candidate_validation_execution" / tmp_path.name


def configured_execution_fixture(tmp_path: Path) -> dict:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    base = repo_relative_tmp(tmp_path)
    config["screening_pipeline"]["screening_benchmark"]["benchmark_name"] = "pytest_screening"
    config["screening_pipeline"]["screening_benchmark"]["artifact_root"] = str(base / "results")
    config["twin"]["intervention_program_registry"]["local_root"] = str(base / "processed")
    config["validation"]["candidate_validation_registry"]["local_root"] = str(base / "processed")
    config["validation"]["candidate_validation_execution"]["output_root_template"] = str(base / "execution" / "{benchmark_name}")
    config["validation"]["candidate_validation_execution"]["input_roots"]["ligand_root"] = str(base / "ligands")
    config["validation"]["candidate_validation_execution"]["input_roots"]["receptor_box_catalog"] = str(base / "structures" / "receptor_box_catalog.tsv")
    config["screening_pipeline"]["action_ready_registry"]["local_root"] = str(base / "processed")
    config["screening_pipeline"]["drug_target_ready_registry"]["local_root"] = str(base / "processed")
    config["screening_pipeline"]["pkpd_ready_registry"]["local_root"] = str(base / "processed")

    intervention_rows = [
        {
            "intervention_program_uid": "intervention_app",
            "regimen_uid": "regimen_app",
            "drug_program_uid_set": "drug_program_app",
            "target_program_set": "target_program_app",
            "target_program_label_set": "APP",
            "compound_set": "compound_app",
            "route": "oral",
            "schedule": "daily",
            "dose_level_id": "dose_100",
            "dose_multiplier": "1.0",
            "occupancy_constraints": "{}",
            "pkpd_feasibility_score": "0.8",
            "evidence_summary": "{}",
            "selection_reason": "case_study_seed",
            "note": "fixture",
        },
        {
            "intervention_program_uid": "intervention_bace1",
            "regimen_uid": "regimen_bace1",
            "drug_program_uid_set": "drug_program_bace1",
            "target_program_set": "target_program_bace1",
            "target_program_label_set": "BACE1",
            "compound_set": "compound_bace1",
            "route": "oral",
            "schedule": "daily",
            "dose_level_id": "dose_100",
            "dose_multiplier": "1.0",
            "occupancy_constraints": "{}",
            "pkpd_feasibility_score": "0.7",
            "evidence_summary": "{}",
            "selection_reason": "ranked_selection",
            "note": "fixture",
        },
    ]
    write_tsv(
        REPO_ROOT / base / "processed" / "intervention_program" / "intervention_program_catalog.tsv",
        config["schemas"]["intervention_program_fields"],
        intervention_rows,
    )
    regimen_rows = [
        {
            "regimen_uid": "regimen_app",
            "drug_program_uid_set": "drug_program_app",
            "compound_uid_set": "compound_app",
            "target_uid_set": "target_app",
            "target_symbol_set": "APP",
            "regimen_scaffold_set": "scaffold_app",
            "route": "oral",
            "schedule": "daily",
            "dose_level_id": "dose_100",
            "dose_multiplier": "1.0",
            "aggregate_target_evidence_score": "0.9",
            "aggregate_affinity_score": "2.0",
            "clue_reversal_proxy_score": "0.5",
            "pk_feasibility_score": "0.8",
            "regimen_size": "1",
            "selection_reason": "case_study_seed",
            "note": "fixture",
        },
        {
            "regimen_uid": "regimen_bace1",
            "drug_program_uid_set": "drug_program_bace1",
            "compound_uid_set": "compound_bace1",
            "target_uid_set": "target_bace1",
            "target_symbol_set": "BACE1",
            "regimen_scaffold_set": "scaffold_bace1",
            "route": "oral",
            "schedule": "daily",
            "dose_level_id": "dose_100",
            "dose_multiplier": "1.0",
            "aggregate_target_evidence_score": "0.8",
            "aggregate_affinity_score": "3.0",
            "clue_reversal_proxy_score": "0.4",
            "pk_feasibility_score": "0.7",
            "regimen_size": "1",
            "selection_reason": "ranked_selection",
            "note": "fixture",
        },
    ]
    write_tsv(
        REPO_ROOT / base / "processed" / "action_ready" / "regimen_catalog.tsv",
        config["schemas"]["action_ready_regimen_fields"],
        regimen_rows,
    )
    write_tsv(
        REPO_ROOT / base / "processed" / "action_ready" / "target_program_catalog.tsv",
        config["schemas"]["action_ready_target_program_fields"],
        [
            {"target_program_uid": "target_program_app", "canonical_entity_id": "GENE_APP", "canonical_entity_label": "APP", "atlas_entity_uid": "atlas_app", "atlas_name": "fixture", "program_id": "program_app", "mapped_feature_ids": "GENE_APP", "mapped_feature_count": "1", "coverage_fraction": "1.0", "target_evidence_score": "0.9", "target_class": "enzyme", "source_repository": "drug_target_direct", "note": ""},
            {"target_program_uid": "target_program_bace1", "canonical_entity_id": "GENE_BACE1", "canonical_entity_label": "BACE1", "atlas_entity_uid": "atlas_bace1", "atlas_name": "fixture", "program_id": "program_bace1", "mapped_feature_ids": "GENE_BACE1", "mapped_feature_count": "1", "coverage_fraction": "1.0", "target_evidence_score": "0.8", "target_class": "enzyme", "source_repository": "drug_target_direct", "note": ""},
        ],
    )
    compound_rows = [
        {
            "compound_uid": "compound_app",
            "canonical_compound_id": "CHEMBL_APP",
            "canonical_id_namespace": "chembl_id",
            "preferred_name": "APP_FIXTURE",
            "canonical_smiles": "CCO",
            "inchi_key": "",
            "murcko_scaffold": "",
            "source_count": "2",
            "source_names": "fixture",
            "chembl_id": "CHEMBL_APP",
            "drugcentral_struct_id": "",
            "bindingdb_ligand_id": "",
            "note": "",
        },
        {
            "compound_uid": "compound_bace1",
            "canonical_compound_id": "CHEMBL_BACE1",
            "canonical_id_namespace": "chembl_id",
            "preferred_name": "BACE1_FIXTURE",
            "canonical_smiles": "CCN",
            "inchi_key": "",
            "murcko_scaffold": "",
            "source_count": "1",
            "source_names": "fixture",
            "chembl_id": "CHEMBL_BACE1",
            "drugcentral_struct_id": "",
            "bindingdb_ligand_id": "",
            "note": "",
        },
    ]
    write_tsv(
        REPO_ROOT / base / "processed" / "drug_target_ready" / "compound_catalog.tsv",
        config["schemas"]["drug_target_compound_fields"],
        compound_rows,
    )
    write_tsv(
        REPO_ROOT / base / "processed" / "drug_target_ready" / "target_catalog.tsv",
        config["schemas"]["drug_target_target_fields"],
        [
            {"target_uid": "target_app", "canonical_entity_id": "GENE_APP", "canonical_entity_label": "APP", "target_symbol": "APP", "target_name": "APP", "target_class": "enzyme", "tractability_bucket": "approved_drug", "source_repository": "fixture", "mapping_method": "exact", "mapping_confidence": "exact", "source_count": "1", "public_url": "", "note": ""},
            {"target_uid": "target_bace1", "canonical_entity_id": "GENE_BACE1", "canonical_entity_label": "BACE1", "target_symbol": "BACE1", "target_name": "BACE1", "target_class": "enzyme", "tractability_bucket": "approved_drug", "source_repository": "fixture", "mapping_method": "exact", "mapping_confidence": "exact", "source_count": "1", "public_url": "", "note": ""},
        ],
    )
    write_tsv(
        REPO_ROOT / base / "processed" / "drug_target_ready" / "compound_feature_catalog.tsv",
        config["schemas"]["drug_target_compound_feature_fields"],
        [
            {"compound_uid": "compound_app", "feature_name": "molecular_weight", "feature_value": "46.0", "feature_type": "numeric_descriptor", "source_repository": "fixture", "note": ""},
            {"compound_uid": "compound_app", "feature_name": "alogp", "feature_value": "1.2", "feature_type": "numeric_descriptor", "source_repository": "fixture", "note": ""},
            {"compound_uid": "compound_app", "feature_name": "tpsa", "feature_value": "20.0", "feature_type": "numeric_descriptor", "source_repository": "fixture", "note": ""},
            {"compound_uid": "compound_app", "feature_name": "hba", "feature_value": "1.0", "feature_type": "numeric_descriptor", "source_repository": "fixture", "note": ""},
            {"compound_uid": "compound_app", "feature_name": "hbd", "feature_value": "1.0", "feature_type": "numeric_descriptor", "source_repository": "fixture", "note": ""},
        ],
    )
    write_tsv(
        REPO_ROOT / base / "processed" / "pkpd_ready" / "compound_pk_prior_catalog.tsv",
        config["schemas"]["pkpd_compound_prior_fields"],
        [
            {"compound_uid": "compound_app", "canonical_compound_id": "CHEMBL_APP", "route": "oral", "permeability_score": "0.8", "exposure_scale": "1.0", "clearance_scale": "0.4", "source_repository": "fixture", "note": ""},
            {"compound_uid": "compound_bace1", "canonical_compound_id": "CHEMBL_BACE1", "route": "oral", "permeability_score": "0.7", "exposure_scale": "0.9", "clearance_scale": "0.5", "source_repository": "fixture", "note": ""},
        ],
    )
    write_tsv(
        REPO_ROOT / base / "processed" / "pkpd_ready" / "parameter_prior_catalog.tsv",
        config["schemas"]["pkpd_parameter_prior_fields"],
        [
            {"parameter_uid": "p1", "parameter_name": "plasma_to_bbb_rate", "compartment_id": "bbb_interface", "unit": "relative_rate", "lower_bound": "0.01", "upper_bound": "1.0", "distribution_family": "uniform", "central_value": "0.01", "source_repository": "fixture", "provenance_level": "fixture", "note": ""},
            {"parameter_uid": "p2", "parameter_name": "bbb_to_csf_rate", "compartment_id": "csf", "unit": "relative_rate", "lower_bound": "0.01", "upper_bound": "1.0", "distribution_family": "uniform", "central_value": "0.01", "source_repository": "fixture", "provenance_level": "fixture", "note": ""},
            {"parameter_uid": "p3", "parameter_name": "csf_to_pvs_rate", "compartment_id": "pvs", "unit": "relative_rate", "lower_bound": "0.01", "upper_bound": "1.0", "distribution_family": "uniform", "central_value": "0.01", "source_repository": "fixture", "provenance_level": "fixture", "note": ""},
            {"parameter_uid": "p4", "parameter_name": "pvs_to_isf_rate", "compartment_id": "isf", "unit": "relative_rate", "lower_bound": "0.01", "upper_bound": "1.0", "distribution_family": "uniform", "central_value": "0.01", "source_repository": "fixture", "provenance_level": "fixture", "note": ""},
            {"parameter_uid": "p5", "parameter_name": "csf_to_meningeal_clearance_rate", "compartment_id": "meningeal_drainage", "unit": "relative_rate", "lower_bound": "0.01", "upper_bound": "1.0", "distribution_family": "uniform", "central_value": "0.01", "source_repository": "fixture", "provenance_level": "fixture", "note": ""},
        ],
    )
    priority_path = REPO_ROOT / base / "results" / "pytest_screening" / "candidate_priority_catalog.tsv"
    write_tsv(
        priority_path,
        config["schemas"]["screening_candidate_priority_fields"],
        [
            {"benchmark_name": "pytest_screening", "fold_name": "fold_alpha", "modality": "transcriptomics", "regimen_uid": "regimen_bace1", "predicted_score_mean": "0.9", "tri_score_mean": "0.8", "pkpd_consistency_mean": "0.7", "predictive_entropy_mean": "0.1", "mutual_information_mean": "0.2", "epistemic_std_mean": "0.3", "active_learning_uncertainty": "0.4", "artifact_path": "fixture", "note": "intervention_bace1"}
        ],
    )
    receptor_path = REPO_ROOT / base / "structures" / "app_receptor.pdbqt"
    receptor_path.parent.mkdir(parents=True, exist_ok=True)
    receptor_path.write_text("RECEPTOR\n", encoding="utf-8")
    write_tsv(
        REPO_ROOT / base / "structures" / "receptor_box_catalog.tsv",
        ["target_label", "structure_source", "receptor_asset_path", "center_x", "center_y", "center_z", "size_x", "size_y", "size_z"],
        [
            {"target_label": "APP", "structure_source": "experimental", "receptor_asset_path": str(base / "structures" / "app_receptor.pdbqt"), "center_x": "1.0", "center_y": "2.0", "center_z": "3.0", "size_x": "10.0", "size_y": "11.0", "size_z": "12.0"}
        ],
    )
    return config


def test_candidate_validation_execution_dry_run_writes_commands_and_pbpk(tmp_path: Path) -> None:
    config = configured_execution_fixture(tmp_path)
    rows = build_execution_rows(config, dry_run=True)
    catalog_path = write_candidate_validation_execution_catalog(config, rows)
    assert catalog_path.exists()
    assert rows
    app_row = next(row for row in rows if row["intervention_program_uid"] == "intervention_app")
    assert app_row["execution_status"] == "planned"
    assert "vina" in app_row["docking_command"]
    assert "gnina" in app_row["rescoring_command"]
    payload = json.loads(str(app_row["pbpk_summary"]))
    assert "brain_plasma_auc_ratio" in payload
    assert float(app_row["pbpk_readiness_score"]) >= 0.0


def test_candidate_validation_docking_stage_reuses_prep_artifacts(tmp_path: Path) -> None:
    config = configured_execution_fixture(tmp_path)
    prep_rows = build_execution_rows(config, dry_run=True, execution_stage="prep")
    assert prep_rows
    docking_rows = build_execution_rows(config, dry_run=True, execution_stage="docking")
    app_row = next(row for row in docking_rows if row["intervention_program_uid"] == "intervention_app")
    assert app_row["execution_status"] == "planned"
    assert app_row["pbpk_artifact_path"]


def test_candidate_validation_execution_missing_binary_is_explicit(tmp_path: Path) -> None:
    config = configured_execution_fixture(tmp_path)
    config["validation"]["candidate_validation_execution"]["binaries"]["vina"] = "definitely_missing_vina_binary"
    with pytest.raises(FileNotFoundError):
        resolve_required_binaries(config)


def test_parse_gnina_score_prefers_cnnaffinity(tmp_path: Path) -> None:
    log_path = tmp_path / "gnina.log"
    log_path.write_text(
        "\n".join(
            [
                "Affinity: -3.47532 (kcal/mol)",
                "CNNscore: 0.27544",
                "CNNaffinity: 3.11843",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert parse_gnina_score(log_path) == pytest.approx(3.11843)


def test_candidate_validation_rows_ingest_execution_artifacts(tmp_path: Path) -> None:
    config = configured_execution_fixture(tmp_path)
    rows = build_execution_rows(config, dry_run=True)
    write_candidate_validation_execution_catalog(config, rows)
    validation_rows = candidate_validation_rows(config)
    app_row = next(row for row in validation_rows if row["intervention_program_uid"] == "intervention_app")
    assert app_row["structure_confidence"] == "1.0"
    assert app_row["docking_score"] == ""
    assert app_row["rescoring_score"] == ""
    assert json.loads(str(app_row["pbpk_summary"]))["brain_plasma_auc_ratio"] >= 0.0
    assert float(app_row["pbpk_readiness_score"]) >= 0.0
    assert app_row["validation_stage"] == "planned"
