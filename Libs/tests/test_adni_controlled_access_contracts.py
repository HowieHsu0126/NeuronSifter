from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
import sys

import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.models.networks.screening_system import ADScreeningSystem
from Libs.exp.training.train_screening_system import rollout_model_kwargs
from Libs.scripts.builders.build_adni_controlled_registry import build_adni_registry_artifacts
from Libs.exp.evaluation.run_adni_controlled_state_inference_benchmark import build_reports
from Libs.scripts.data_acquisition.sync_adni_controlled_access import (
    apply_collection_first_profile_override,
    build_authenticated_session,
    build_browser_cookie_session,
    build_password_login_session,
    collection_candidate_links,
    discovered_asset_rows,
    infer_manifest_columns,
    load_adni_credentials,
    manifest_first_asset_rows,
    manifest_schema_audit,
    redacted_manifest_dry_run_summary,
    read_ida_export_manifest,
    redacted_collection_first_summary,
    run_collection_first_no_download_dry_run,
    run_manifest_first_no_download_dry_run,
    sync_adni_assets,
)
from Libs.scripts.validators.validate_adni_controlled_access import validate_adni_outputs
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, sha256_of_file, write_tsv
from Libs.utils.model.dynamics_fields import node_adjacency_matrix
from Libs.utils.model.screening_model_config import screening_model_kwargs_from_config


def configured_adni_test_stack(tmp_path: Path) -> dict:
    config = deepcopy(load_config(REPO_ROOT / "Libs/configs/config_stack_controlled_research.yaml"))
    private_root = tmp_path / "private" / "adni"
    processed_root = tmp_path / "processed" / "adni"
    public_reports = tmp_path / "public_reports"
    public_logs = tmp_path / "public_logs"
    config["logging_registry"]["console_enabled"] = False
    config["logging_registry"]["file_enabled"] = False
    config["adni_controlled_access"]["paths"]["private_root"] = str(private_root)
    config["adni_controlled_access"]["paths"]["processed_root"] = str(processed_root)
    config["adni_controlled_access"]["paths"]["state_inference_summary"] = str(public_reports / "adni_state_inference_summary.json")
    config["adni_controlled_access"]["paths"]["longitudinal_validation_summary"] = str(public_reports / "adni_longitudinal_validation_summary.json")
    config["adni_controlled_access"]["paths"]["missing_modality_robustness"] = str(public_reports / "adni_missing_modality_robustness.json")
    config["adni_controlled_access"]["paths"]["world_state_factor_anchoring"] = str(public_reports / "adni_world_state_factor_anchoring.json")
    config["adni_controlled_access"]["paths"]["posterior_calibration_summary"] = str(public_reports / "adni_posterior_calibration_summary.json")
    config["adni_controlled_access"]["paths"]["controlled_bundle_pointer"] = str(public_reports / "adni_controlled_bundle_pointer.json")
    config["adni_controlled_access"]["acquisition"]["mode"] = "approved_asset_manifest"
    config["adni_controlled_access"]["builder"]["split_policy"]["train_fraction"] = 1.0
    config["adni_controlled_access"]["builder"]["split_policy"]["validation_fraction"] = 0.0
    config["adni_controlled_access"]["builder"]["split_policy"]["test_fraction"] = 0.0
    config["adni_controlled_access"]["application"]["adni_tabular_adapter"]["inference_splits"] = ["train"]
    config["adni_controlled_access"]["validators"]["public_scan_roots"] = [
        str(public_reports),
        str(public_logs),
    ]
    ensure_directory(public_reports)
    ensure_directory(public_logs)
    return config


def write_synthetic_formal_mainline_checkpoint(config: dict, tmp_path: Path) -> None:
    checkpoint_config = config["adni_controlled_access"]["application"]["screening_checkpoint"]
    artifact_root = tmp_path / "formal_mainline"
    seed_value = int(checkpoint_config["seed_value"])
    benchmark_name = str(checkpoint_config["benchmark_name_template"]).format(seed=seed_value)
    checkpoint_root = artifact_root / benchmark_name / "production" / "ad_screening_system"
    ensure_directory(checkpoint_root)
    rollout_kwargs = rollout_model_kwargs(config)
    model_kwargs = {
        **screening_model_kwargs_from_config(config),
        "patient_state_dim": 12,
        "target_feature_dim": 4,
        "regimen_feature_dim": 32,
        "hidden_dim": 16,
        "latent_dim": 8,
        "dropout_rate": 0.0,
        "compartment_count": len(config["screening_pipeline"]["pkpd_ready_registry"]["compartments"]),
        "tri_weight_by_component": dict(config["screening_pipeline"]["screening_benchmark"]["tri_weights"]),
        "adjacency_matrix": node_adjacency_matrix(config).tolist(),
        "rollout_step_count": int(rollout_kwargs["rollout_step_count"]),
        "fast_steps_per_slow_step": int(rollout_kwargs["fast_steps_per_slow_step"]),
        "fast_state_indices": list(rollout_kwargs["fast_state_indices"]),
        "slow_state_indices": list(rollout_kwargs["slow_state_indices"]),
        "slow_update_scale": float(rollout_kwargs["slow_update_scale"]),
        "rollout_update_policy": dict(rollout_kwargs["rollout_update_policy"]),
    }
    model = ADScreeningSystem(**model_kwargs)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_kwargs": model_kwargs,
            "model_contract_version": str(checkpoint_config["expected_model_contract_version"]),
        },
        checkpoint_root / "model_weights.pt",
    )
    (checkpoint_root / "training_manifest.json").write_text(
        json.dumps(
            {
                "scope": "production",
                "split_name": "production",
                "fold_name": "production",
                "seed": seed_value,
                "ensemble_member_seed": seed_value,
                "model_name": "ad_screening_system",
                "benchmark_name": benchmark_name,
                "model_contract_version": str(checkpoint_config["expected_model_contract_version"]),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    checkpoint_config["artifact_root"] = str(artifact_root)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_synthetic_governance_documents(config: dict) -> None:
    private_root = Path(config["adni_controlled_access"]["paths"]["private_root"])
    evidence_config = config["adni_controlled_access"]["governance"]["evidence"]
    governance_root = private_root / evidence_config["governance_root"]
    ensure_directory(governance_root)
    for document_config in evidence_config["required_documents"]:
        document_path = governance_root / document_config["relative_path"]
        ensure_directory(document_path.parent)
        document_path.write_text(
            "\n".join(
                [
                    f"document_id: {document_config['document_id']}",
                    "scope: controlled validation calibration",
                    "redistribution: participant level redistribution prohibited",
                ]
            ),
            encoding="utf-8",
        )


def write_synthetic_adni_exports(tmp_path: Path) -> dict[str, Path]:
    export_root = tmp_path / "authorized_exports"
    clinical_path = export_root / "ADNIMERGE.csv"
    diagnosis_path = export_root / "ADNI_DX.csv"
    demographics_path = export_root / "PTDEMOG.csv"
    cognitive_path = export_root / "NEUROBAT.csv"
    medications_path = export_root / "RECMHIST.csv"
    neuropsychiatric_path = export_root / "NPIQ.csv"
    apoe_path = export_root / "APOERES.csv"
    mri_freesurfer_path = export_root / "UCSFFSL.csv"
    imaging_path = export_root / "t1_inventory.csv"
    amyloid_path = export_root / "amyloid_pet.csv"
    tau_path = export_root / "tau_pet.csv"
    fdg_path = export_root / "fdg_pet.csv"
    csf_path = export_root / "csf_biomarkers.csv"
    plasma_path = export_root / "plasma_biomarkers.csv"
    omics_path = export_root / "omics_inventory.csv"
    clinical_fields = [
        "RID",
        "PTID",
        "VISCODE",
        "EXAMDATE",
        "DX",
        "AGE",
        "PTGENDER",
        "PTEDUCAT",
        "MMSE",
        "CDRSB",
        "ADAS13",
        "FAQ",
        "NPI",
        "Hippocampus",
        "Entorhinal",
        "WholeBrain",
        "ICV",
        "AV45",
        "TAU_SUVR",
        "FDG",
        "ABETA",
        "PTAU",
        "TAU",
        "NFL",
        "GFAP",
        "COLPROT",
        "SITEID",
    ]
    write_csv(
        clinical_path,
        clinical_fields,
        [
            {
                "RID": "1",
                "PTID": "001_S_0001",
                "VISCODE": "bl",
                "EXAMDATE": "2020-01-01",
                "DX": "CN",
                "AGE": "70",
                "PTGENDER": "Female",
                "PTEDUCAT": "16",
                "MMSE": "29",
                "CDRSB": "0.5",
                "ADAS13": "8",
                "FAQ": "1",
                "NPI": "0",
                "Hippocampus": "6500",
                "Entorhinal": "3.5",
                "WholeBrain": "1000000",
                "ICV": "1500000",
                "AV45": "1.1",
                "TAU_SUVR": "1.2",
                "FDG": "1.3",
                "ABETA": ">900",
                "PTAU": "20",
                "TAU": "300",
                "NFL": "12",
                "GFAP": "100",
                "COLPROT": "ADNI1",
                "SITEID": "001",
            },
            {
                "RID": "1",
                "PTID": "001_S_0001",
                "VISCODE": "m12",
                "EXAMDATE": "2021-01-01",
                "DX": "MCI",
                "AGE": "71",
                "PTGENDER": "Female",
                "PTEDUCAT": "16",
                "MMSE": "27",
                "CDRSB": "1.0",
                "ADAS13": "12",
                "FAQ": "2",
                "NPI": "1",
                "Hippocampus": "6200",
                "Entorhinal": "3.2",
                "WholeBrain": "990000",
                "ICV": "1500000",
                "AV45": "1.3",
                "TAU_SUVR": "1.4",
                "FDG": "1.1",
                "ABETA": "820",
                "PTAU": "24",
                "TAU": "330",
                "NFL": "14",
                "GFAP": "110",
                "COLPROT": "ADNI1",
                "SITEID": "001",
            },
            {
                "RID": "2",
                "PTID": "002_S_0002",
                "VISCODE": "bl",
                "EXAMDATE": "2020-02-01",
                "DX": "AD",
                "AGE": "75",
                "PTGENDER": "Male",
                "PTEDUCAT": "14",
                "MMSE": "22",
                "CDRSB": "4.0",
                "ADAS13": "26",
                "FAQ": "10",
                "NPI": "3",
                "Hippocampus": "5400",
                "Entorhinal": "2.7",
                "WholeBrain": "940000",
                "ICV": "1480000",
                "AV45": "1.8",
                "TAU_SUVR": "1.9",
                "FDG": "0.9",
                "ABETA": "500",
                "PTAU": "35",
                "TAU": "450",
                "NFL": "20",
                "GFAP": "150",
                "COLPROT": "ADNI2",
                "SITEID": "002",
            },
        ],
    )
    visit_rows = [
        {"RID": "1", "PTID": "001_S_0001", "VISCODE": "bl", "DX": "CN", "COLPROT": "ADNI1", "SITEID": "001"},
        {"RID": "1", "PTID": "001_S_0001", "VISCODE": "m12", "DX": "MCI", "COLPROT": "ADNI1", "SITEID": "001"},
        {"RID": "2", "PTID": "002_S_0002", "VISCODE": "bl", "DX": "AD", "COLPROT": "ADNI2", "SITEID": "002"},
    ]
    write_csv(diagnosis_path, ["RID", "PTID", "VISCODE", "DX", "COLPROT", "SITEID"], visit_rows)
    write_csv(
        demographics_path,
        ["RID", "PTID", "VISCODE", "AGE", "PTGENDER", "PTEDUCAT", "COLPROT", "SITEID"],
        [
            {"RID": "1", "PTID": "001_S_0001", "VISCODE": "bl", "AGE": "70", "PTGENDER": "Female", "PTEDUCAT": "16", "COLPROT": "ADNI1", "SITEID": "001"},
            {"RID": "1", "PTID": "001_S_0001", "VISCODE": "m12", "AGE": "71", "PTGENDER": "Female", "PTEDUCAT": "16", "COLPROT": "ADNI1", "SITEID": "001"},
            {"RID": "2", "PTID": "002_S_0002", "VISCODE": "bl", "AGE": "75", "PTGENDER": "Male", "PTEDUCAT": "14", "COLPROT": "ADNI2", "SITEID": "002"},
        ],
    )
    write_csv(
        cognitive_path,
        ["RID", "PTID", "VISCODE", "MMSE", "CDRSB", "ADAS13", "FAQ", "NPI"],
        [
            {"RID": "1", "PTID": "001_S_0001", "VISCODE": "bl", "MMSE": "29", "CDRSB": "0.5", "ADAS13": "8", "FAQ": "1", "NPI": "0"},
            {"RID": "1", "PTID": "001_S_0001", "VISCODE": "m12", "MMSE": "27", "CDRSB": "1.0", "ADAS13": "12", "FAQ": "2", "NPI": "1"},
            {"RID": "2", "PTID": "002_S_0002", "VISCODE": "bl", "MMSE": "22", "CDRSB": "4.0", "ADAS13": "26", "FAQ": "10", "NPI": "3"},
        ],
    )
    write_csv(
        medications_path,
        ["RID", "PTID", "VISCODE", "CMTRT", "CMINDC", "CMSTDTC", "CMENDTC"],
        [
            {"RID": "1", "PTID": "001_S_0001", "VISCODE": "bl", "CMTRT": "donepezil", "CMINDC": "recorded", "CMSTDTC": "2020-01-01", "CMENDTC": ""},
            {"RID": "1", "PTID": "001_S_0001", "VISCODE": "m12", "CMTRT": "memantine", "CMINDC": "recorded", "CMSTDTC": "2021-01-01", "CMENDTC": ""},
            {"RID": "2", "PTID": "002_S_0002", "VISCODE": "bl", "CMTRT": "rivastigmine", "CMINDC": "recorded", "CMSTDTC": "2020-02-01", "CMENDTC": ""},
        ],
    )
    write_csv(
        neuropsychiatric_path,
        ["RID", "PTID", "VISCODE", "NPISCORE"],
        [
            {"RID": "1", "PTID": "001_S_0001", "VISCODE": "bl", "NPISCORE": "0"},
            {"RID": "1", "PTID": "001_S_0001", "VISCODE": "m12", "NPISCORE": "1"},
            {"RID": "2", "PTID": "002_S_0002", "VISCODE": "bl", "NPISCORE": "3"},
        ],
    )
    write_csv(
        apoe_path,
        ["RID", "APGEN1", "APGEN2"],
        [
            {"RID": "1", "APGEN1": "3", "APGEN2": "4"},
            {"RID": "2", "APGEN1": "4", "APGEN2": "4"},
        ],
    )
    write_csv(
        mri_freesurfer_path,
        ["RID", "PTID", "VISCODE", "Hippocampus", "Entorhinal", "WholeBrain", "ICV"],
        [
            {"RID": "1", "PTID": "001_S_0001", "VISCODE": "bl", "Hippocampus": "6500", "Entorhinal": "3.5", "WholeBrain": "1000000", "ICV": "1500000"},
            {"RID": "1", "PTID": "001_S_0001", "VISCODE": "m12", "Hippocampus": "6200", "Entorhinal": "3.2", "WholeBrain": "990000", "ICV": "1500000"},
            {"RID": "2", "PTID": "002_S_0002", "VISCODE": "bl", "Hippocampus": "5400", "Entorhinal": "2.7", "WholeBrain": "940000", "ICV": "1480000"},
        ],
    )
    write_csv(
        imaging_path,
        ["RID", "PTID", "VISCODE", "MODALITY", "IMAGEUID", "SEQUENCE", "COLLECTION", "LOCAL_PATH", "CHECKSUM", "SCANDATE", "SITEID", "COLPROT"],
        [
            {
                "RID": "1",
                "PTID": "001_S_0001",
                "VISCODE": "bl",
                "MODALITY": "MRI T1",
                "IMAGEUID": "I1001",
                "SEQUENCE": "MPRAGE",
                "COLLECTION": "ADNI MRI",
                "LOCAL_PATH": "Input/private/adni/downloads/images/adni_image_a.nii.gz",
                "CHECKSUM": "sha256-a",
                "SCANDATE": "2020-01-01",
                "SITEID": "001",
                "COLPROT": "ADNI1",
            },
            {
                "RID": "1",
                "PTID": "001_S_0001",
                "VISCODE": "m12",
                "MODALITY": "Amyloid PET",
                "IMAGEUID": "I1002",
                "SEQUENCE": "AV45",
                "COLLECTION": "ADNI PET",
                "LOCAL_PATH": "Input/private/adni/downloads/images/adni_image_b.nii.gz",
                "CHECKSUM": "sha256-b",
                "SCANDATE": "2021-01-01",
                "SITEID": "001",
                "COLPROT": "ADNI1",
            },
        ],
    )
    write_csv(
        amyloid_path,
        ["RID", "PTID", "VISCODE", "AV45"],
        [
            {"RID": "1", "PTID": "001_S_0001", "VISCODE": "bl", "AV45": "1.1"},
            {"RID": "1", "PTID": "001_S_0001", "VISCODE": "m12", "AV45": "1.3"},
            {"RID": "2", "PTID": "002_S_0002", "VISCODE": "bl", "AV45": "1.8"},
        ],
    )
    write_csv(
        tau_path,
        ["RID", "PTID", "VISCODE", "TAU_SUVR"],
        [
            {"RID": "1", "PTID": "001_S_0001", "VISCODE": "bl", "TAU_SUVR": "1.2"},
            {"RID": "1", "PTID": "001_S_0001", "VISCODE": "m12", "TAU_SUVR": "1.4"},
            {"RID": "2", "PTID": "002_S_0002", "VISCODE": "bl", "TAU_SUVR": "1.9"},
        ],
    )
    write_csv(
        fdg_path,
        ["RID", "PTID", "VISCODE", "FDG"],
        [
            {"RID": "1", "PTID": "001_S_0001", "VISCODE": "bl", "FDG": "1.3"},
            {"RID": "1", "PTID": "001_S_0001", "VISCODE": "m12", "FDG": "1.1"},
            {"RID": "2", "PTID": "002_S_0002", "VISCODE": "bl", "FDG": "0.9"},
        ],
    )
    write_csv(
        csf_path,
        ["RID", "PTID", "VISCODE", "ABETA", "PTAU", "TAU"],
        [
            {"RID": "1", "PTID": "001_S_0001", "VISCODE": "bl", "ABETA": "900", "PTAU": "20", "TAU": "300"},
            {"RID": "1", "PTID": "001_S_0001", "VISCODE": "m12", "ABETA": "820", "PTAU": "24", "TAU": "330"},
            {"RID": "2", "PTID": "002_S_0002", "VISCODE": "bl", "ABETA": "500", "PTAU": "35", "TAU": "450"},
        ],
    )
    write_csv(
        plasma_path,
        ["RID", "PTID", "VISCODE", "NFL", "GFAP"],
        [
            {"RID": "1", "PTID": "001_S_0001", "VISCODE": "bl", "NFL": "12", "GFAP": "100"},
            {"RID": "1", "PTID": "001_S_0001", "VISCODE": "m12", "NFL": "14", "GFAP": "110"},
            {"RID": "2", "PTID": "002_S_0002", "VISCODE": "bl", "NFL": "20", "GFAP": "150"},
        ],
    )
    write_csv(
        omics_path,
        ["RID", "PTID", "VISCODE", "INVENTORY_UID", "MODALITY", "ASSAY", "AVAILABLE"],
        [
            {"RID": "1", "PTID": "001_S_0001", "VISCODE": "bl", "INVENTORY_UID": "omics-a", "MODALITY": "GENOMICS", "ASSAY": "APOE inventory", "AVAILABLE": "yes"},
            {"RID": "2", "PTID": "002_S_0002", "VISCODE": "bl", "INVENTORY_UID": "omics-b", "MODALITY": "GENOMICS", "ASSAY": "APOE inventory", "AVAILABLE": "yes"},
        ],
    )
    return {
        "adni_merge_core": clinical_path,
        "visit_diagnosis": diagnosis_path,
        "demographics": demographics_path,
        "cognitive_scores": cognitive_path,
        "neuropsychiatric_scores": neuropsychiatric_path,
        "medications": medications_path,
        "apoe_genetics": apoe_path,
        "mri_freesurfer": mri_freesurfer_path,
        "t1_inventory": imaging_path,
        "amyloid_pet": amyloid_path,
        "tau_pet": tau_path,
        "fdg_pet": fdg_path,
        "csf_biomarkers": csf_path,
        "plasma_biomarkers": plasma_path,
        "omics_inventory": omics_path,
    }


def write_approved_asset_manifest(config: dict, export_paths: dict[str, Path]) -> None:
    private_root = Path(config["adni_controlled_access"]["paths"]["private_root"])
    manifest_path = private_root / config["adni_controlled_access"]["paths"]["approved_asset_manifest"]
    role_registry = config["adni_controlled_access"]["sync"]["role_registry"]
    rows = []
    for table_role in config["adni_controlled_access"]["sync"]["required_table_roles"]:
        role_config = role_registry[table_role]
        rows.append(
            {
                "asset_uid": f"adni_{table_role}",
                "include_flag": "yes",
                "transport_kind": "local_authorized_export",
                "asset_kind": role_config["asset_kind"],
                "data_domain": role_config["data_domain"],
                "table_role": table_role,
                "modality": role_config["modality"],
                "adni_cohort": "ADNI1_ADNI2",
                "source_url": "",
                "source_path": export_paths[table_role].as_posix(),
                "target_relative_path": role_config["target_relative_path"],
                "participant_level_flag": "yes",
                "allowed_use": "validation_internal_only",
                "notes": "synthetic contract table",
            }
        )
    write_tsv(manifest_path, config["adni_controlled_access"]["sync"]["approved_asset_manifest_fields"], rows)


def prepare_synced_synthetic_adni(config: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    export_paths = write_synthetic_adni_exports(tmp_path)
    write_synthetic_governance_documents(config)
    write_approved_asset_manifest(config, export_paths)
    monkeypatch.setenv("ADNI_IDA_EMAIL", "researcher@example.edu")
    monkeypatch.setenv("ADNI_IDA_PASSWORD", "not_logged_secret")
    sync_adni_assets(config)
    private_root = Path(config["adni_controlled_access"]["paths"]["private_root"])
    raw_manifest_path = private_root / config["adni_controlled_access"]["paths"]["raw_manifest"]
    raw_manifest = pd.read_csv(raw_manifest_path, sep="\t", dtype=str, keep_default_na=False)
    assert set(raw_manifest["table_role"]) == set(config["adni_controlled_access"]["sync"]["required_table_roles"])
    for row in raw_manifest.to_dict(orient="records"):
        assert row["checksum_sha256"] == sha256_of_file(REPO_ROOT / row["local_path_private"])


def test_adni_controlled_access_config_preserves_public_boundary() -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack_controlled_research.yaml")
    adni_config = config["adni_controlled_access"]
    assert config["source_policy"]["strict_public_only_flag"] is True
    assert config["source_policy"]["imaging_enabled_flag"] is False
    assert adni_config["governance"]["publication_boundary"] == "preserve_public_only_mainline"
    assert adni_config["paths"]["private_root"] == "Input/private/adni"
    assert adni_config["paths"]["processed_root"] == "Input/processed/adni"
    assert adni_config["credentials"]["email_env"] == "ADNI_IDA_EMAIL"
    assert adni_config["credentials"]["password_env"] == "ADNI_IDA_PASSWORD"
    assert adni_config["credentials"]["hash_salt_env"] == "ADNI_HASH_SALT"
    assert adni_config["governance"]["evidence"]["governance_root"] == "ADNI_access_governance"
    assert len(adni_config["governance"]["evidence"]["required_documents"]) == 5
    assert adni_config["acquisition"]["mode"] == "collection_first"
    assert adni_config["acquisition"]["minimum_downloadable_records"] == 1
    assert adni_config["acquisition"]["collection_first"]["build_manifest_only"] is True
    assert adni_config["acquisition"]["collection_first"]["download_enabled"] is False
    assert adni_config["acquisition"]["collection_first"]["enforce_required_roles_on_download"] is True
    assert adni_config["acquisition"]["collection_first"]["max_records"] == 512
    assert adni_config["acquisition"]["collection_first"]["browser_engine"] == "auto"
    assert "--no-proxy-server" in set(adni_config["acquisition"]["collection_first"]["playwright_launch_args"])
    assert adni_config["acquisition"]["collection_first"]["deny_raw_dicom_or_full_image"] is True
    assert adni_config["acquisition"]["collection_first"]["arc_builder"]["enabled"] is True
    assert adni_config["acquisition"]["collection_first"]["arc_builder"]["download_tables_path"] == "/ajax2/xml/search/downloadTables_v2"
    assert "adni_merge_core" in adni_config["acquisition"]["collection_first"]["arc_builder"]["role_study_file_patterns"]
    assert "visit_diagnosis" in adni_config["acquisition"]["collection_first"]["arc_builder"]["role_table_categories"]
    assert "neuropsychiatric_scores" in adni_config["acquisition"]["collection_first"]["arc_builder"]["role_table_categories"]
    assert adni_config["acquisition"]["manifest_first"]["generated_manifest_path"] == "Input/private/adni/manifests/ida_generated_manifest.tsv"
    assert adni_config["acquisition"]["collection_first"]["execution_profile"] == "build_manifest_only"
    assert adni_config["acquisition"]["collection_first"]["profiles"]["build_manifest_only"]["download_enabled"] is False
    assert adni_config["acquisition"]["collection_first"]["profiles"]["smoke_download"]["max_records"] == 1
    assert adni_config["acquisition"]["collection_first"]["profiles"]["required_roles_download"]["download_enabled"] is True
    assert adni_config["sync"]["authenticated_http"]["authentication_order"] == "browser_cookie_jar_first"
    assert adni_config["sync"]["authenticated_http"]["password_login_fallback"] is False
    assert adni_config["sync"]["authenticated_http"]["cookie_jar_format"] == "netscape_mozilla"
    assert adni_config["sync"]["authenticated_http"]["cookie_jar_path"] == "Input/private/adni/secrets/ida.loni.usc.edu_cookies.txt"
    assert "Input/private/adni/secrets/cookies.txt" in set(adni_config["sync"]["authenticated_http"]["candidate_cookie_jar_paths"])
    assert "ida.loni.usc.edu" in set(adni_config["sync"]["authenticated_http"]["cookie_domain_allowlist"])
    assert "IDA_USC" in set(adni_config["sync"]["authenticated_http"]["cookie_policy"]["allowed_session_cookie_names"])
    assert adni_config["sync"]["authenticated_http"]["cookie_policy"]["accept_zero_expiry_session_cookies"] is True
    assert adni_config["sync"]["ida_discovery"]["required_gate"] is False
    assert {"adni_merge_core", "mri_freesurfer", "plasma_biomarkers", "omics_inventory"}.issubset(
        set(adni_config["sync"]["required_table_roles"])
    )
    assert adni_config["application"]["model_family"] == "screening_checkpoint_world_state_adapter_with_train_split_centroid_calibrator"
    assert adni_config["application"]["screening_checkpoint"]["required"] is True
    assert adni_config["application"]["screening_checkpoint"]["selection_policy"] == "current_public_mainline_production"
    assert adni_config["application"]["screening_checkpoint"]["expected_scope"] == "production"
    assert adni_config["application"]["screening_checkpoint"]["expected_fold_name"] == "production"
    assert adni_config["application"]["screening_checkpoint"]["state_dict_strict"] is True
    assert adni_config["application"]["adni_tabular_adapter"]["adapter_name"] == "train_split_state_projection_to_screening_world_state"


def test_adni_sync_requires_env_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = configured_adni_test_stack(tmp_path)
    export_paths = write_synthetic_adni_exports(tmp_path)
    write_approved_asset_manifest(config, export_paths)
    monkeypatch.delenv("ADNI_IDA_EMAIL", raising=False)
    monkeypatch.delenv("ADNI_IDA_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="controlled-access form-login credentials"):
        load_adni_credentials(config)


class FakeCookie:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeCookies(list):
    def set(self, name: str, value: str, domain: str | None = None, path: str | None = None) -> None:
        self.append(FakeCookie(name))

    def update(self, cookie_jar: object) -> None:
        for cookie in cookie_jar:
            self.append(FakeCookie(cookie.name))


class FakeAdniResponse:
    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP status {self.status_code}")

    def close(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        payload = self.text.encode("utf-8")
        for offset in range(0, len(payload), chunk_size):
            yield payload[offset:offset + chunk_size]


class FakeAdniSession:
    def __init__(self) -> None:
        self.headers: dict[str, str] = {}
        self.cookies = FakeCookies([FakeCookie("JSESSIONID")])
        self.post_payload: dict[str, str] = {}

    def get(self, url: str, **_: object) -> FakeAdniResponse:
        if "login.jsp" in url:
            return FakeAdniResponse('<input type="hidden" name="project" value="ADNI"><input name="userEmail"><input name="userPassword">')
        return FakeAdniResponse(
            """
            <a href="/downloads/ADNIMERGE.csv">ADNIMERGE</a>
            <a href="/downloads/APOERES.csv">APOE</a>
            <a href="/downloads/UCSFFSL.csv">FreeSurfer</a>
            """
        )

    def post(self, _: str, data: dict[str, str]) -> FakeAdniResponse:
        self.post_payload = dict(data)
        return FakeAdniResponse("success")


def test_adni_ida_login_and_discovery_use_configured_selectors(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_config(REPO_ROOT / "Libs/configs/config_stack_controlled_research.yaml")
    config["logging_registry"]["console_enabled"] = False
    config["logging_registry"]["file_enabled"] = False
    fake_session = FakeAdniSession()
    monkeypatch.setattr("Libs.scripts.data_acquisition.sync_adni_controlled_access.requests.Session", lambda: fake_session)
    session = build_password_login_session(config, {"email": "researcher@example.edu", "password": "secret"})
    rows = discovered_asset_rows(config, session)
    assert fake_session.post_payload["project"] == "ADNI"
    assert fake_session.post_payload["userEmail"] == "researcher@example.edu"
    assert fake_session.post_payload["isAjax"] == "true"
    assert {"adni_merge_core", "apoe_genetics", "mri_freesurfer"}.issubset({row["table_role"] for row in rows})
    assert all(row["transport_kind"] == "authenticated_http" for row in rows)


def write_netscape_cookie_jar(cookie_path: Path) -> None:
    ensure_directory(cookie_path.parent)
    cookie_path.write_text(
        "\n".join(
            [
                "# Netscape HTTP Cookie File",
                ".ida.loni.usc.edu\tTRUE\t/\tFALSE\t1893456000\tJSESSIONID\tbrowser_session_value",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_non_ida_cookie_jar(cookie_path: Path) -> None:
    ensure_directory(cookie_path.parent)
    cookie_path.write_text(
        "\n".join(
            [
                "# Netscape HTTP Cookie File",
                ".example.org\tTRUE\t/\tFALSE\t1893456000\tNOT_IDA\tbrowser_session_value",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_adni_browser_cookie_jar_loads_and_skips_password_login(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = configured_adni_test_stack(tmp_path)
    cookie_path = tmp_path / "private" / "adni" / "secrets" / "ida.loni.usc.edu_cookies.txt"
    config["adni_controlled_access"]["sync"]["authenticated_http"]["cookie_jar_path"] = str(cookie_path)
    write_netscape_cookie_jar(cookie_path)
    fake_session = FakeAdniSession()
    monkeypatch.setattr("Libs.scripts.data_acquisition.sync_adni_controlled_access.requests.Session", lambda: fake_session)
    monkeypatch.delenv("ADNI_IDA_EMAIL", raising=False)
    monkeypatch.delenv("ADNI_IDA_PASSWORD", raising=False)

    session = build_authenticated_session(config)

    assert session is fake_session
    assert fake_session.post_payload == {}
    assert "JSESSIONID" in {cookie.name for cookie in fake_session.cookies}


def test_adni_browser_cookie_jar_uses_declared_candidate_without_password_login(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = configured_adni_test_stack(tmp_path)
    primary_cookie_path = tmp_path / "private" / "adni" / "secrets" / "ida.loni.usc.edu_cookies.txt"
    alternate_cookie_path = tmp_path / "private" / "adni" / "secrets" / "cookies.txt"
    config["adni_controlled_access"]["sync"]["authenticated_http"]["cookie_jar_path"] = str(primary_cookie_path)
    config["adni_controlled_access"]["sync"]["authenticated_http"]["candidate_cookie_jar_paths"] = [str(alternate_cookie_path)]
    write_non_ida_cookie_jar(primary_cookie_path)
    write_netscape_cookie_jar(alternate_cookie_path)
    fake_session = FakeAdniSession()
    monkeypatch.setattr("Libs.scripts.data_acquisition.sync_adni_controlled_access.requests.Session", lambda: fake_session)
    monkeypatch.setattr(
        "Libs.scripts.data_acquisition.sync_adni_controlled_access.build_password_login_session",
        lambda *_: pytest.fail("password login fallback must not run"),
    )

    session = build_authenticated_session(config)

    assert session is fake_session
    assert getattr(session, "_adni_cookie_jar_path") == alternate_cookie_path
    assert fake_session.post_payload == {}


class FakeExpiredCookieSession(FakeAdniSession):
    def get(self, url: str, **_: object) -> FakeAdniResponse:
        if "search.jsp" in url:
            return FakeAdniResponse('<input name="userEmail"><input name="userPassword">')
        return super().get(url)


class FakeNoRoleDiscoverySession(FakeAdniSession):
    def get(self, url: str, **_: object) -> FakeAdniResponse:
        if "search.jsp" in url:
            return FakeAdniResponse("<html><body>Study Data download portal</body></html>")
        return super().get(url)


class FakeCollectionFirstSession(FakeAdniSession):
    def get(self, url: str, **_: object) -> FakeAdniResponse:
        if "getSearches_v2" in url:
            return FakeAdniResponse(
                """
                <?xml version="1.0" encoding="UTF-8"?>
                <searches>
                  <search id="-1" name="ADNI Recommended PET and MRI" loaded="false">
                    <search id="1" name="ADNI Recommended PET and MRI" loaded="true"/>
                  </search>
                </searches>
                """
            )
        if "getVariables_v2" in url:
            return FakeAdniResponse(
                """
                <?xml version="1.0" encoding="UTF-8"?>
                <variables>
                  <group name="PTDEMOG" description="Subject Demographics"><variable id="54" code="RID"/></group>
                  <group name="DXSUM" description="Diagnostic Summary"><variable id="855" code="RID"/></group>
                  <group name="NEUROBAT" description="Neuropsychological Battery"><variable id="115" code="RID"/></group>
                  <group name="NPIQ" description="Neuropsychiatric Inventory Questionnaire"><variable id="1489" code="RID"/></group>
                  <group name="RECMHIST" description="Recent Medical History Details Log"><variable id="995" code="RID"/></group>
                  <group name="APOERES" description="ApoE Genotyping"><variable id="109" code="RID"/></group>
                  <group name="UCSFFSX7" description="UCSF FreeSurfer"><variable id="2901" code="RID"/></group>
                  <group name="MRI3META" description="MRI Scan Information"><variable id="1777" code="RID"/></group>
                  <group name="UCBERKELEY_AMY_6MM" description="Amyloid PET"><variable id="3240" code="RID"/></group>
                  <group name="UCBERKELEY_TAU_6MM" description="Tau PET"><variable id="3901" code="RID"/></group>
                  <group name="UPENNBIOMK_ROCHE_ELECSYS" description="CSF Biomarkers"><variable id="771" code="RID"/></group>
                  <group name="UPENN_PLASMA_FUJIREBIO_QUANTERIX" description="Plasma Biomarkers"><variable id="2050" code="RID"/></group>
                  <group name="GENETIC" description="Genetic Sample Collection"><variable id="103" code="RID"/></group>
                </variables>
                """
            )
        if "getStudyCategories_v2" in url:
            return FakeAdniResponse(
                """
                <?xml version="1.0" encoding="UTF-8"?>
                <study>
                  <category id="1" name="_Archive">
                    <subcategory id="1" name="Archived Data">
                      <item id="453" name="ADNIMERGE - Key ADNI tables merged into one table [ADNI1,GO,2,3]" type="table" fileSize="13259788"/>
                      <item id="2117" name="UC Berkeley - FDG Analysis [ADNI1,GO,2,3]" type="table" fileSize="398322"/>
                    </subcategory>
                  </category>
                </study>
                """
            )
        if "collection_manifest.csv" in url:
            return FakeAdniResponse(
                "\n".join(
                    [
                        "ImageUID,Download URL,Modality,Visit,Subject,Phase,File Name,File Size",
                        "I1001,https://ida.loni.usc.edu/download/image?id=I1001&token=SECRET_TOKEN_VALUE,Clinical,bl,001_S_0001,ADNI1,ADNIMERGE.csv,12345",
                    ]
                )
            )
        if "advancedDownload" in url:
            return FakeAdniResponse(
                """
                <html><title>Advanced Download</title>
                <a href="/download/file?token=ADVANCED_SECRET">1-Click Download</a>
                </html>
                """
            )
        if "search.jsp" in url or "login.jsp" in url:
            return FakeAdniResponse(
                """
                <html><title>IDA Data Collections</title>
                <form action="/pages/access/search.jsp?csrf=PRIVATE_FORM_TOKEN" method="POST"></form>
                <a href="/pages/access/collection_manifest.csv?token=CSV_SECRET">CSV</a>
                <a href="/pages/access/advancedDownload.jsp?token=ADV_SECRET">Advanced Download</a>
                <a href="/pages/access/search.jsp?page=DOWNLOADS&subPage=DATA_COLLECTIONS&token=PRIVATE">Data Collections</a>
                </html>
                """
            )
        return super().get(url)

    def post(self, url: str, data: dict[str, str]) -> FakeAdniResponse:
        self.post_payload = dict(data)
        if "downloadTables_v2" in url:
            return FakeAdniResponse(
                f"""
                <?xml version="1.0" encoding="UTF-8"?>
                <download><path name="{data.get('name', 'arc_table')}.csv"/></download>
                """
            )
        if "getStudyData" in url:
            file_id = str(data.get("fileId", "study_file"))
            return FakeAdniResponse(
                f"""
                <?xml version="1.0" encoding="UTF-8"?>
                <studyData><path name="{file_id}.csv"/></studyData>
                """
            )
        return super().post(url, data)


def configure_manifest_first_cookie_stack(config: dict, tmp_path: Path, manifest_path: Path) -> Path:
    cookie_path = tmp_path / "private" / "adni" / "secrets" / "ida.loni.usc.edu_cookies.txt"
    config["adni_controlled_access"]["acquisition"]["mode"] = "manifest_first"
    config["adni_controlled_access"]["acquisition"]["manifest_path"] = str(manifest_path)
    config["adni_controlled_access"]["sync"]["authenticated_http"]["cookie_jar_path"] = str(cookie_path)
    write_netscape_cookie_jar(cookie_path)
    return cookie_path


def configure_collection_first_cookie_stack(config: dict, tmp_path: Path) -> Path:
    generated_manifest = tmp_path / "private" / "adni" / "manifests" / "ida_generated_manifest.tsv"
    cookie_path = tmp_path / "private" / "adni" / "secrets" / "ida.loni.usc.edu_cookies.txt"
    config["adni_controlled_access"]["acquisition"]["mode"] = "collection_first"
    config["adni_controlled_access"]["acquisition"]["manifest_path"] = str(generated_manifest)
    config["adni_controlled_access"]["acquisition"]["collection_first"]["build_manifest_only"] = True
    config["adni_controlled_access"]["acquisition"]["collection_first"]["download_enabled"] = False
    config["adni_controlled_access"]["sync"]["authenticated_http"]["cookie_jar_path"] = str(cookie_path)
    write_netscape_cookie_jar(cookie_path)
    return generated_manifest


def write_ida_manifest_schema_one(path: Path) -> None:
    write_csv(
        path,
        ["ImageUID", "Download URL", "Modality", "Visit", "Subject", "Phase", "File Name", "File Size"],
        [
            {
                "ImageUID": "I1001",
                "Download URL": "https://ida.loni.usc.edu/download/image?id=I1001&token=SECRET_TOKEN_VALUE",
                "Modality": "MRI",
                "Visit": "bl",
                "Subject": "001_S_0001",
                "Phase": "ADNI1",
                "File Name": "I1001_MRI.nii",
                "File Size": "12345",
            }
        ],
    )


def write_ida_manifest_schema_two(path: Path) -> None:
    ensure_directory(path.parent)
    path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "fileId": "F2001",
                        "packageId": "P100",
                        "scan_type": "PET",
                        "VISCODE": "m12",
                        "participant_id": "002_S_0002",
                        "COLPROT": "ADNI2",
                        "archive_name": "F2001_PET.dcm",
                        "bytes": "54321",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def test_adni_browser_cookie_jar_fail_fast_without_password_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = configured_adni_test_stack(tmp_path)
    cookie_path = tmp_path / "private" / "adni" / "secrets" / "ida.loni.usc.edu_cookies.txt"
    config["adni_controlled_access"]["sync"]["authenticated_http"]["cookie_jar_path"] = str(cookie_path)
    config["adni_controlled_access"]["sync"]["authenticated_http"]["candidate_cookie_jar_paths"] = []
    write_netscape_cookie_jar(cookie_path)
    fake_session = FakeExpiredCookieSession()
    monkeypatch.setattr("Libs.scripts.data_acquisition.sync_adni_controlled_access.requests.Session", lambda: fake_session)
    monkeypatch.setattr(
        "Libs.scripts.data_acquisition.sync_adni_controlled_access.build_password_login_session",
        lambda *_: pytest.fail("password login fallback must not run"),
    )

    with pytest.raises(RuntimeError, match="browser cookie jar"):
        build_browser_cookie_session(config)


def test_adni_manifest_first_valid_cookie_and_manifest_skip_password_login(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = configured_adni_test_stack(tmp_path)
    manifest_path = tmp_path / "private" / "adni" / "manifests" / "ida_export_manifest.csv"
    configure_manifest_first_cookie_stack(config, tmp_path, manifest_path)
    write_ida_manifest_schema_one(manifest_path)
    fake_session = FakeNoRoleDiscoverySession()
    monkeypatch.setattr("Libs.scripts.data_acquisition.sync_adni_controlled_access.requests.Session", lambda: fake_session)
    monkeypatch.setattr(
        "Libs.scripts.data_acquisition.sync_adni_controlled_access.build_password_login_session",
        lambda *_: pytest.fail("password login fallback must not run"),
    )
    monkeypatch.delenv("ADNI_IDA_EMAIL", raising=False)
    monkeypatch.delenv("ADNI_IDA_PASSWORD", raising=False)

    audit = run_manifest_first_no_download_dry_run(config)

    assert fake_session.post_payload == {}
    assert audit["cookie_session_status"] == "authenticated_cookie_session_valid"
    assert audit["direct_download_record_count"] == 1
    assert audit["downloadable_record_count"] == 1
    audit_path = Path(config["adni_controlled_access"]["paths"]["private_root"]) / config["adni_controlled_access"]["paths"]["manifest_schema_audit"]
    assert audit_path.exists()


def test_adni_manifest_first_missing_manifest_fails_fast(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = configured_adni_test_stack(tmp_path)
    manifest_path = tmp_path / "private" / "adni" / "manifests" / "missing_manifest.tsv"
    configure_manifest_first_cookie_stack(config, tmp_path, manifest_path)
    fake_session = FakeNoRoleDiscoverySession()
    monkeypatch.setattr("Libs.scripts.data_acquisition.sync_adni_controlled_access.requests.Session", lambda: fake_session)

    with pytest.raises(FileNotFoundError, match="ADNI acquisition manifest is missing"):
        run_manifest_first_no_download_dry_run(config)


def test_adni_manifest_first_allows_zero_role_discovery_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = configured_adni_test_stack(tmp_path)
    manifest_path = tmp_path / "private" / "adni" / "manifests" / "ida_export_manifest.csv"
    configure_manifest_first_cookie_stack(config, tmp_path, manifest_path)
    write_ida_manifest_schema_one(manifest_path)
    fake_session = FakeNoRoleDiscoverySession()
    monkeypatch.setattr("Libs.scripts.data_acquisition.sync_adni_controlled_access.requests.Session", lambda: fake_session)

    session = build_authenticated_session(config)
    discovery_rows = discovered_asset_rows(config, session)
    audit = run_manifest_first_no_download_dry_run(config)

    assert discovery_rows == []
    assert audit["downloadable_record_count"] == 1


def test_adni_manifest_parser_supports_csv_and_json_fixture_schemas(tmp_path: Path) -> None:
    csv_manifest = tmp_path / "schema_one.csv"
    json_manifest = tmp_path / "schema_two.json"
    write_ida_manifest_schema_one(csv_manifest)
    write_ida_manifest_schema_two(json_manifest)
    csv_rows = pd.read_csv(csv_manifest, dtype=str, keep_default_na=False).to_dict(orient="records")
    json_rows = json.loads(json_manifest.read_text(encoding="utf-8"))["records"]
    config = configured_adni_test_stack(tmp_path)
    config["adni_controlled_access"]["acquisition"]["mode"] = "manifest_first"
    config["adni_controlled_access"]["acquisition"]["manifest_path"] = str(csv_manifest)

    csv_inferred = infer_manifest_columns(csv_rows)
    json_inferred = infer_manifest_columns(json_rows)
    csv_audit = manifest_schema_audit(config, csv_rows, "authenticated_cookie_session_valid")

    assert {"ImageUID"}.issubset(set(csv_inferred["file_id"]))
    assert {"Download URL"}.issubset(set(csv_inferred["download_url"]))
    assert {"fileId"}.issubset(set(json_inferred["file_id"]))
    assert {"packageId"}.issubset(set(json_inferred["package_id"]))
    assert {"archive_name"}.issubset(set(json_inferred["filename"]))
    assert csv_audit["downloadable_record_count"] == 1


def test_adni_manifest_records_map_to_required_roles(tmp_path: Path) -> None:
    config = configured_adni_test_stack(tmp_path)
    manifest_path = tmp_path / "private" / "adni" / "manifests" / "required_tables.csv"
    config["adni_controlled_access"]["acquisition"]["mode"] = "manifest_first"
    config["adni_controlled_access"]["acquisition"]["manifest_path"] = str(manifest_path)
    write_csv(
        manifest_path,
        ["Download URL", "File Name", "Modality", "Phase"],
        [
            {
                "Download URL": "https://ida.loni.usc.edu/download/table?file=ADNIMERGE.csv&token=SECRET_TOKEN_VALUE",
                "File Name": "ADNIMERGE.csv",
                "Modality": "clinical",
                "Phase": "ADNI",
            }
        ],
    )
    rows = read_ida_export_manifest(config)
    audit = manifest_schema_audit(config, rows, "authenticated_cookie_session_valid")

    asset_rows = manifest_first_asset_rows(config, rows, audit)

    assert asset_rows[0]["asset_uid"] == "adni_adni_merge_core"
    assert asset_rows[0]["table_role"] == "adni_merge_core"
    assert asset_rows[0]["asset_kind"] == "clinical_table"
    assert asset_rows[0]["data_domain"] == "clinical_longitudinal"
    assert asset_rows[0]["target_relative_path"] == "downloads/tables/ADNIMERGE.csv"


def test_adni_manifest_dry_run_summary_redacts_sensitive_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    config = configured_adni_test_stack(tmp_path)
    manifest_path = tmp_path / "private" / "adni" / "manifests" / "ida_export_manifest.csv"
    configure_manifest_first_cookie_stack(config, tmp_path, manifest_path)
    write_ida_manifest_schema_one(manifest_path)
    fake_session = FakeNoRoleDiscoverySession()
    monkeypatch.setattr("Libs.scripts.data_acquisition.sync_adni_controlled_access.requests.Session", lambda: fake_session)

    audit = run_manifest_first_no_download_dry_run(config)
    print(json.dumps(redacted_manifest_dry_run_summary(audit), sort_keys=True))
    captured = capsys.readouterr()

    assert "browser_session_value" not in captured.out
    assert "SECRET_TOKEN_VALUE" not in captured.out
    assert "password" not in captured.out.lower()
    assert "001_S_0001" not in captured.out


def test_adni_collection_first_generates_manifest_without_password_login(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = configured_adni_test_stack(tmp_path)
    generated_manifest = configure_collection_first_cookie_stack(config, tmp_path)
    fake_session = FakeCollectionFirstSession()
    monkeypatch.setattr("Libs.scripts.data_acquisition.sync_adni_controlled_access.requests.Session", lambda: fake_session)
    monkeypatch.setattr(
        "Libs.scripts.data_acquisition.sync_adni_controlled_access.build_password_login_session",
        lambda *_: pytest.fail("password login fallback must not run"),
    )

    audit = run_collection_first_no_download_dry_run(config)
    generated_rows = read_ida_export_manifest(config)

    assert "userEmail" not in fake_session.post_payload
    assert "userPassword" not in fake_session.post_payload
    assert generated_manifest.exists()
    assert audit["acquisition_mode"] == "collection_first"
    assert audit["generated_manifest_record_count"] >= len(config["adni_controlled_access"]["sync"]["required_table_roles"])
    assert len(generated_rows) >= len(config["adni_controlled_access"]["sync"]["required_table_roles"])
    assert any(str(row.get("download_url", "")).startswith("https://ida.loni.usc.edu/download/files/") for row in generated_rows)


def test_adni_collection_first_arc_builder_covers_required_roles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = configured_adni_test_stack(tmp_path)
    configure_collection_first_cookie_stack(config, tmp_path)
    fake_session = FakeCollectionFirstSession()
    monkeypatch.setattr("Libs.scripts.data_acquisition.sync_adni_controlled_access.requests.Session", lambda: fake_session)
    monkeypatch.setattr(
        "Libs.scripts.data_acquisition.sync_adni_controlled_access.build_password_login_session",
        lambda *_: pytest.fail("password login fallback must not run"),
    )

    audit = run_collection_first_no_download_dry_run(config)
    generated_rows = read_ida_export_manifest(config)
    generated_roles = {str(row.get("table_role", "")) for row in generated_rows}

    assert set(config["adni_controlled_access"]["sync"]["required_table_roles"]).issubset(generated_roles)
    assert audit["manifest_schema_audit"]["missing_required_role_count"] == 0
    assert audit["manifest_schema_audit"]["matched_required_role_count"] == len(config["adni_controlled_access"]["sync"]["required_table_roles"])
    assert "userPassword" not in fake_session.post_payload


def test_adni_collection_first_required_roles_profile_downloads_without_password_login(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = configured_adni_test_stack(tmp_path)
    configure_collection_first_cookie_stack(config, tmp_path)
    apply_collection_first_profile_override(config, "required_roles_download")
    fake_session = FakeCollectionFirstSession()
    monkeypatch.setattr("Libs.scripts.data_acquisition.sync_adni_controlled_access.requests.Session", lambda: fake_session)
    monkeypatch.setattr(
        "Libs.scripts.data_acquisition.sync_adni_controlled_access.build_password_login_session",
        lambda *_: pytest.fail("password login fallback must not run"),
    )

    receipt_rows = sync_adni_assets(config)

    required_roles = set(config["adni_controlled_access"]["sync"]["required_table_roles"])
    assert {str(row["table_role"]) for row in receipt_rows} == required_roles
    assert {str(row["status"]) for row in receipt_rows} == {"downloaded"}
    assert "userEmail" not in fake_session.post_payload
    assert "userPassword" not in fake_session.post_payload
    for row in receipt_rows:
        assert Path(row["local_path_private"]).exists()


def test_adni_collection_first_role_discovery_zero_does_not_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = configured_adni_test_stack(tmp_path)
    configure_collection_first_cookie_stack(config, tmp_path)
    fake_session = FakeCollectionFirstSession()
    monkeypatch.setattr("Libs.scripts.data_acquisition.sync_adni_controlled_access.requests.Session", lambda: fake_session)

    session = build_authenticated_session(config)
    discovery_rows = discovered_asset_rows(config, session)
    audit = run_collection_first_no_download_dry_run(config)

    assert discovery_rows == []
    assert audit["generated_manifest_record_count"] >= len(config["adni_controlled_access"]["sync"]["required_table_roles"])


def test_adni_collection_first_logs_redacted_inventory_and_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = configured_adni_test_stack(tmp_path)
    configure_collection_first_cookie_stack(config, tmp_path)
    fake_session = FakeCollectionFirstSession()
    monkeypatch.setattr("Libs.scripts.data_acquisition.sync_adni_controlled_access.requests.Session", lambda: fake_session)

    audit = run_collection_first_no_download_dry_run(config)
    print(json.dumps(redacted_collection_first_summary(audit), sort_keys=True))
    captured = capsys.readouterr()
    private_root = Path(config["adni_controlled_access"]["paths"]["private_root"])
    inventory_path = private_root / config["adni_controlled_access"]["paths"]["collection_page_inventory"]
    inventory_text = inventory_path.read_text(encoding="utf-8")

    assert "browser_session_value" not in captured.out
    assert "SECRET_TOKEN_VALUE" not in captured.out
    assert "password" not in captured.out.lower()
    assert "CSV_SECRET" not in inventory_text
    assert "ADV_SECRET" not in inventory_text
    assert "PRIVATE_FORM_TOKEN" not in inventory_text


def test_adni_collection_first_writes_next_action_report_when_no_collection_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = configured_adni_test_stack(tmp_path)
    configure_collection_first_cookie_stack(config, tmp_path)
    config["adni_controlled_access"]["acquisition"]["collection_first"]["arc_builder"]["enabled"] = False
    config["adni_controlled_access"]["acquisition"]["collection_first"]["browser_engine"] = "requests"
    fake_session = FakeNoRoleDiscoverySession()
    monkeypatch.setattr("Libs.scripts.data_acquisition.sync_adni_controlled_access.requests.Session", lambda: fake_session)

    with pytest.raises(RuntimeError, match="next-action report"):
        run_collection_first_no_download_dry_run(config)

    private_root = Path(config["adni_controlled_access"]["paths"]["private_root"])
    report_path = private_root / config["adni_controlled_access"]["paths"]["collection_next_action_report"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["missing_entry"] == "collection_csv_or_advanced_download"
    assert report["password_login_fallback"] == "disabled"
    assert "Playwright trace" in report["required_next_action"]


def test_adni_sync_fails_fast_when_required_role_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = configured_adni_test_stack(tmp_path)
    config["adni_controlled_access"]["sync"]["ida_discovery"]["enabled"] = False
    export_paths = write_synthetic_adni_exports(tmp_path)
    write_approved_asset_manifest(config, export_paths)
    private_root = Path(config["adni_controlled_access"]["paths"]["private_root"])
    manifest_path = private_root / config["adni_controlled_access"]["paths"]["approved_asset_manifest"]
    rows = pd.read_csv(manifest_path, sep="\t", dtype=str, keep_default_na=False)
    rows = rows[rows["table_role"] != "plasma_biomarkers"]
    rows.to_csv(manifest_path, sep="\t", index=False)
    monkeypatch.setenv("ADNI_IDA_EMAIL", "researcher@example.edu")
    monkeypatch.setenv("ADNI_IDA_PASSWORD", "not_logged_secret")
    with pytest.raises(RuntimeError, match="plasma_biomarkers"):
        sync_adni_assets(config)


def test_adni_builder_fails_fast_when_governance_document_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = configured_adni_test_stack(tmp_path)
    export_paths = write_synthetic_adni_exports(tmp_path)
    write_approved_asset_manifest(config, export_paths)
    monkeypatch.setenv("ADNI_IDA_EMAIL", "researcher@example.edu")
    monkeypatch.setenv("ADNI_IDA_PASSWORD", "not_logged_secret")
    sync_adni_assets(config)
    monkeypatch.setenv("ADNI_HASH_SALT", "synthetic_hash_salt_for_tests")

    with pytest.raises(FileNotFoundError, match="ADNI_access_governance"):
        build_adni_registry_artifacts(config)

    private_root = Path(config["adni_controlled_access"]["paths"]["private_root"])
    governance_report_path = private_root / config["adni_controlled_access"]["paths"]["access_governance_report"]
    governance_report = json.loads(governance_report_path.read_text(encoding="utf-8"))
    assert governance_report["document_status"] == "missing_required_documents"
    assert governance_report["missing_required_document_paths"]


def test_adni_sync_builder_and_validator_redact_participant_identifiers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = configured_adni_test_stack(tmp_path)
    # The synthetic legacy adapter supplies a projected sample tensor.
    # Select its existing compatibility posterior for this redaction control.
    config["screening_pipeline"]["screening_benchmark"]["model_architecture"][
        "world_state_posterior"
    ]["family"] = "deterministic_compatibility"
    application = config["adni_controlled_access"]["application"]
    application["screening_checkpoint"]["seed_value"] = 41
    application["adni_tabular_adapter"]["projection_policy"]["random_seed"] = 41
    prepare_synced_synthetic_adni(config, tmp_path, monkeypatch)
    imaging_path = Path(config["adni_controlled_access"]["paths"]["private_root"]) / "downloads" / "imaging" / "t1_inventory.csv"
    write_csv(
        imaging_path,
        ["PHASE", "PTID", "RID", "VISCODE", "ID", "SITEID"],
        [
            {"PHASE": "ADNI1", "PTID": "001_S_0001", "RID": "1", "VISCODE": "bl", "ID": "MRI3META_1", "SITEID": "001"},
            {"PHASE": "ADNI1", "PTID": "001_S_0001", "RID": "1", "VISCODE": "m12", "ID": "MRI3META_2", "SITEID": "001"},
        ],
    )
    raw_manifest_path = Path(config["adni_controlled_access"]["paths"]["private_root"]) / config["adni_controlled_access"]["paths"]["raw_manifest"]
    raw_manifest = pd.read_csv(raw_manifest_path, sep="\t", dtype=str, keep_default_na=False)
    imaging_mask = raw_manifest["table_role"] == "t1_inventory"
    raw_manifest.loc[imaging_mask, "checksum_sha256"] = sha256_of_file(imaging_path)
    raw_manifest.loc[imaging_mask, "size_bytes"] = str(imaging_path.stat().st_size)
    raw_manifest.to_csv(raw_manifest_path, sep="\t", index=False)
    monkeypatch.setenv("ADNI_HASH_SALT", "synthetic_hash_salt_for_tests")
    output_paths = build_adni_registry_artifacts(config)
    output_paths["controlled_bundle_pointer"].unlink()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        write_synthetic_formal_mainline_checkpoint(config, tmp_path)
        report_paths = build_reports(config)
    validate_adni_outputs(config)
    captured = capsys.readouterr()
    assert "not_logged_secret" not in captured.out
    assert "not_logged_secret" not in captured.err
    processed_root = Path(config["adni_controlled_access"]["paths"]["processed_root"])
    processed_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [
            output_paths["sample_observation_registry"],
            output_paths["longitudinal_transition_table"],
            output_paths["imaging_inventory"],
        ]
    )
    assert "001_S_0001" not in processed_text
    assert "002_S_0002" not in processed_text
    sample_frame = pd.read_csv(output_paths["sample_observation_registry"], sep="\t", dtype=str, keep_default_na=False)
    feature_frame = pd.read_parquet(output_paths["feature_matrix"])
    split_audit = pd.read_json(processed_root / config["adni_controlled_access"]["paths"]["split_audit"], typ="series")
    assert len(sample_frame) == 3
    assert "benchmark_split" in sample_frame.columns
    assert sample_frame.groupby("subject_uid_hash")["benchmark_split"].nunique().max() == 1
    assert {"mmse_total", "hippocampus_volume", "amyloid_suvr", "apoe4_count"}.issubset(set(feature_frame["feature_name"]))
    assert int(split_audit["subject_holdout_leakage_count"]) == 0
    assert "modality_missingness_by_split" in split_audit
    assert output_paths["matrix_norm"].exists()
    assert output_paths["train_scaler"].exists()
    assert output_paths["access_governance_report"].exists()
    assert output_paths["controlled_bundle_pointer"].exists()
    assert report_paths["state_inference_summary"].exists()
    assert report_paths["world_state_factor_anchoring"].exists()
    assert report_paths["posterior_calibration_summary"].exists()
    assert report_paths["world_state_predictions"].exists()
    governance_report = json.loads(output_paths["access_governance_report"].read_text(encoding="utf-8"))
    bundle_pointer = json.loads(output_paths["controlled_bundle_pointer"].read_text(encoding="utf-8"))
    state_summary = json.loads(report_paths["state_inference_summary"].read_text(encoding="utf-8"))
    posterior_summary = json.loads(report_paths["posterior_calibration_summary"].read_text(encoding="utf-8"))
    prediction_frame = pd.read_parquet(report_paths["world_state_predictions"])
    assert governance_report["document_status"] == "complete"
    assert bundle_pointer["participant_level_flag"] == "aggregate_only"
    assert set(bundle_pointer["role_counts"]) == set(config["adni_controlled_access"]["sync"]["required_table_roles"])
    assert state_summary["participant_level_flag"] == "aggregate_only"
    assert posterior_summary["participant_level_flag"] == "aggregate_only"
    assert list(prediction_frame.columns) == config["schemas"]["adni_world_state_prediction_fields"]


def test_adni_validator_rejects_subject_split_leakage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = configured_adni_test_stack(tmp_path)
    prepare_synced_synthetic_adni(config, tmp_path, monkeypatch)
    monkeypatch.setenv("ADNI_HASH_SALT", "synthetic_hash_salt_for_tests")
    output_paths = build_adni_registry_artifacts(config)
    sample_frame = pd.read_csv(output_paths["sample_observation_registry"], sep="\t", dtype=str, keep_default_na=False)
    repeated_subject = sample_frame["subject_uid_hash"].value_counts().idxmax()
    repeated_index = sample_frame.index[sample_frame["subject_uid_hash"] == repeated_subject][-1]
    sample_frame.loc[repeated_index, "benchmark_split"] = "test" if sample_frame.loc[repeated_index, "benchmark_split"] != "test" else "train"
    sample_frame.to_csv(output_paths["sample_observation_registry"], sep="\t", index=False)
    with pytest.raises(ValueError, match="subject-holdout leakage"):
        validate_adni_outputs(config)
