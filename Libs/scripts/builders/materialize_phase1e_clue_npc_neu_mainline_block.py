"""Materialize the accession-scoped CLUE NPC/NEU perturbation M1 block."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
import tempfile
from typing import Mapping

import h5py
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.public_foundation_corpus import canonical_sha256, hash_path
from Libs.utils.data.public_foundation_evidence import sha256_file
from Libs.utils.data.public_foundation_roles import upsert_accession_requalification_delta
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, write_json
from Libs.utils.evaluation.alignment_benchmark import write_parquet_atomic
from Libs.utils.model.dynamics_store import initialize_zarr_store, write_zarr_array_from_loader


BUILDER_ID = "Libs/scripts/builders/materialize_phase1e_clue_npc_neu_mainline_block.py"
MANIFEST_VERSION = "neuronsifter-phase1e-clue-npc-neu-perturbation-block-v1"
AUDIT_VERSION = "neuronsifter-g2-m1-perturbation-audit-v1"
CANONICAL_UID = "clue_touchstone_v1_1_npc_neu_perturbation_phase1e_v1"
MATRIX_DATASET = "/0/DATA/0/matrix"
MATRIX_COLUMN_IDS = "/0/META/COL/id"
MATRIX_ROW_IDS = "/0/META/ROW/id"


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        value = " ".join(data.split())
        if value:
            self.parts.append(value)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON payload must be an object: {path}")
    return payload


def _binding(path: Path, *, known_sha256: str | None = None) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": _relative(path),
        "bytes": path.stat().st_size,
        "sha256": known_sha256 or sha256_file(path),
    }


def _record(path: Path, name: str) -> dict[str, object]:
    file_count, byte_count, digest = hash_path(path)
    return {
        "name": name,
        "relative_path": name,
        "file_count": file_count,
        "bytes": byte_count,
        "sha256": digest,
    }


def _content_hash(records: list[Mapping[str, object]]) -> str:
    return canonical_sha256([
        {
            "name": row["name"],
            "file_count": row["file_count"],
            "bytes": row["bytes"],
            "sha256": row["sha256"],
        }
        for row in records
    ])


def _uid(prefix: str, payload: object) -> str:
    return f"{prefix}:{canonical_sha256(payload)[:24]}"


def _decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ], dtype=str)


def _normalized_number(value: object) -> str:
    return format(float(value), ".12g")


def _canonical_physical_unit(value: object) -> str:
    token = str(value).strip().casefold().replace("µ", "u").replace("μ", "u")
    return {"um": "uM", "h": "h"}.get(token, "")


def _terms_text(path: Path) -> str:
    parser = _TextExtractor()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    return " ".join(parser.parts)


def _validate_release_evidence(policy: Mapping[str, object]) -> dict[str, object]:
    candidate = policy["candidate"]
    assert isinstance(candidate, Mapping)
    terms_path = _path("Input/raw/clue_lincs/terms/clue_terms_snapshot.html")
    terms = _terms_text(terms_path).casefold()
    required_terms = {
        "research_group_only": "restricted to use by you within your research group",
        "no_redistribution": "please do not redistribute them",
        "derivative_distribution_requires_contact": "derivative work that is significantly different",
        "publication_allowed": "encourage you to publish results from analyses of these data",
        "nonprofit_academic_only": "restricted to academic use within not-for-profit institutions",
    }
    missing = [key for key, phrase in required_terms.items() if phrase not in terms]
    if missing:
        raise ValueError("CLUE terms snapshot lacks required role evidence: " + ",".join(missing))

    build_detail = _load_json(_path("Input/raw/clue_lincs/downloads/builds/TouchstoneV1.1/build_detail.json"))
    terms_of_use = build_detail.get("data_terms_of_use")
    if not isinstance(terms_of_use, list) or not any(
        isinstance(row, Mapping) and row.get("name") == "published" for row in terms_of_use
    ):
        raise ValueError("Touchstone build detail is not marked as a published dataset")
    if build_detail.get("status") != "Completed" or build_detail.get("name") != "TouchstoneV1.1":
        raise ValueError("Touchstone release identity is not complete")

    receipt_path = _path("Input/raw/clue_lincs/downloads/download_receipts.tsv")
    with receipt_path.open("r", encoding="utf-8", newline="") as handle:
        receipts = list(csv.DictReader(handle, delimiter="\t"))
    archive_path = str(candidate["raw_archive_path"])
    receipt = next((row for row in receipts if row.get("local_path") == archive_path), None)
    if receipt is None or receipt.get("status") != "downloaded" or receipt.get("asset_url") != candidate["official_value_url"]:
        raise ValueError("Official Touchstone archive retrieval receipt is absent or mismatched")
    if receipt.get("download_date") != candidate["retrieval_date"]:
        raise ValueError("Touchstone retrieval date does not match the frozen receipt")

    checksum_path = _path("Input/raw/clue_lincs/checksums/sha256.tsv")
    with checksum_path.open("r", encoding="utf-8", newline="") as handle:
        checksum_rows = list(csv.DictReader(handle, delimiter="\t", fieldnames=["path", "sha256", "bytes"]))
    clue_root = _path("Input/raw/clue_lincs")
    declared = {str((clue_root / row["path"]).resolve()): row for row in checksum_rows}
    for path_key, hash_key, bytes_key in (
        ("raw_value_path", "expected_raw_sha256", "expected_raw_bytes"),
        ("raw_archive_path", "expected_archive_sha256", "expected_archive_bytes"),
    ):
        path = _path(str(candidate[path_key]))
        row = declared.get(str(path))
        if row is None or row["sha256"] != candidate[hash_key] or int(row["bytes"]) != int(candidate[bytes_key]):
            raise ValueError(f"Checksum ledger does not bind {path_key}")
    return {
        "release_status": "qualified_published_release",
        "terms_status": "qualified_internal_nonprofit_research_only",
        "terms_snapshot": _binding(terms_path),
        "release_detail": _binding(_path("Input/raw/clue_lincs/downloads/builds/TouchstoneV1.1/build_detail.json")),
        "download_receipt": _binding(receipt_path),
        "checksum_ledger": _binding(checksum_path),
        "license_role": policy["license_role"],
    }


def _matrix_layout(path: Path) -> tuple[tuple[int, int], np.dtype, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        matrix = handle[MATRIX_DATASET]
        shape = (int(matrix.shape[0]), int(matrix.shape[1]))
        dtype = np.dtype(matrix.dtype)
        column_ids = _decode(np.asarray(handle[MATRIX_COLUMN_IDS]))
        row_ids = _decode(np.asarray(handle[MATRIX_ROW_IDS]))
    if shape != (len(column_ids), len(row_ids)):
        raise ValueError("GCTX matrix and metadata axes disagree")
    return shape, dtype, column_ids, row_ids


def _feature_table(gene_path: Path, row_ids: np.ndarray) -> pd.DataFrame:
    genes = pd.read_csv(gene_path, sep="\t", dtype=str, keep_default_na=False)
    if genes["pr_gene_id"].duplicated().any():
        raise ValueError("Touchstone gene metadata contains duplicate Entrez IDs")
    indexed = genes.set_index("pr_gene_id", drop=False)
    missing = sorted(set(row_ids) - set(indexed.index))
    if missing:
        raise ValueError(f"Touchstone feature metadata lacks {len(missing)} matrix row IDs")
    ordered = indexed.loc[row_ids].reset_index(drop=True)
    return pd.DataFrame({
        "feature_index": np.arange(len(ordered), dtype=np.int64),
        "feature_namespace": "NCBI_Entrez_Gene",
        "feature_id": "NCBIGene:" + ordered["pr_gene_id"].astype(str),
        "source_feature_id": ordered["pr_gene_id"].astype(str),
        "gene_symbol": ordered["pr_gene_symbol"].astype(str),
        "gene_title": ordered["pr_gene_title"].astype(str),
        "is_landmark": ordered["pr_is_lm"].eq("1"),
        "is_bing": ordered["pr_is_bing"].eq("1"),
    })


def _selection_tables(
    policy: Mapping[str, object],
    column_ids: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    candidate = policy["candidate"]
    qc = policy["qc"]
    assert isinstance(candidate, Mapping) and isinstance(qc, Mapping)
    metadata_paths = [Path(str(value)).name for value in candidate["metadata_paths"]]
    metadata = {name: _path(str(value)) for name, value in zip(metadata_paths, candidate["metadata_paths"])}
    sig = pd.read_csv(metadata["GSE92742_Broad_LINCS_sig_info.txt.gz"], sep="\t", dtype=str, keep_default_na=False)
    inst = pd.read_csv(metadata["GSE92742_Broad_LINCS_inst_info.txt.gz"], sep="\t", dtype=str, keep_default_na=False)
    pert = pd.read_csv(metadata["GSE92742_Broad_LINCS_pert_info.txt.gz"], sep="\t", dtype=str, keep_default_na=False)
    cell = pd.read_csv(metadata["GSE92742_Broad_LINCS_cell_info.txt.gz"], sep="\t", dtype=str, keep_default_na=False)

    eligible_cells = list(map(str, candidate["eligible_cell_ids"]))
    cell_rows = cell[cell["cell_id"].isin(eligible_cells)].copy()
    if set(cell_rows["cell_id"]) != set(eligible_cells):
        raise ValueError("Eligible NPC/NEU context metadata is incomplete")
    npc = cell_rows[cell_rows["cell_id"].eq("NPC")].iloc[0]
    neu = cell_rows[cell_rows["cell_id"].eq("NEU")].iloc[0]
    if (
        npc["cell_type"] != "differentiated"
        or npc["sample_type"] != "primary"
        or npc["primary_site"] != "central nervous system"
        or "iPSC" not in npc["modification"]
    ):
        raise ValueError("NPC is not supported as an iPSC-derived CNS model by source metadata")
    if (
        neu["cell_type"] != "differentiated"
        or "neurons" not in neu["modification"]
        or neu["sample_type"] != "normal"
    ):
        raise ValueError("NEU is not supported as a differentiated neuronal model by source metadata")

    source_subset = sig[
        sig["cell_id"].isin(eligible_cells)
        & sig["pert_type"].eq(str(qc["allowed_perturbation_type"]))
    ].copy()
    source_subset["dose_numeric"] = pd.to_numeric(source_subset["pert_dose"], errors="coerce")
    source_subset["time_numeric"] = pd.to_numeric(source_subset["pert_time"], errors="coerce")
    source_subset["physical_axes_valid"] = (
        source_subset["dose_numeric"].gt(0)
        & source_subset["time_numeric"].gt(0)
        & source_subset["pert_dose_unit"].isin(list(map(str, qc["allowed_raw_dose_units"])))
        & source_subset["pert_time_unit"].isin(list(map(str, qc["allowed_raw_time_units"])))
    )

    all_distil_ids = {
        instance_id
        for value in source_subset.loc[source_subset["physical_axes_valid"], "distil_id"]
        for instance_id in str(value).split("|")
        if instance_id
    }
    treatment_instances = inst[inst["inst_id"].isin(all_distil_ids)].set_index("inst_id", drop=False)
    controls = inst[
        inst["cell_id"].isin(eligible_cells) & inst["pert_type"].eq("ctl_vehicle")
    ].copy()
    controls["time_key"] = controls["pert_time"].map(_normalized_number)
    controls_by_plate_time = {
        (str(cell_id), str(plate), str(time_key), str(time_unit)): tuple(sorted(group["inst_id"].astype(str)))
        for (cell_id, plate, time_key, time_unit), group in controls.groupby(
            ["cell_id", "rna_plate", "time_key", "pert_time_unit"], sort=True
        )
    }
    pert_by_id = pert.set_index("pert_id", drop=False)
    column_index = {value: index for index, value in enumerate(column_ids.tolist())}
    qualified_rows: list[dict[str, object]] = []
    quarantine_rows: list[dict[str, object]] = []

    for source_row in source_subset.to_dict(orient="records"):
        sig_id = str(source_row["sig_id"])
        reason = ""
        if not bool(source_row["physical_axes_valid"]):
            reason = "physical_dose_or_time_invalid"
        distil_ids = tuple(sorted(value for value in str(source_row["distil_id"]).split("|") if value))
        if not reason and len(distil_ids) < int(qc["minimum_source_instance_count"]):
            reason = "source_instance_count_below_minimum"
        instance_rows: list[Mapping[str, object]] = []
        if not reason:
            for instance_id in distil_ids:
                if instance_id not in treatment_instances.index:
                    reason = "source_instance_metadata_missing"
                    break
                instance = treatment_instances.loc[instance_id]
                if isinstance(instance, pd.DataFrame):
                    reason = "source_instance_id_duplicated"
                    break
                instance_rows.append(instance.to_dict())
        if not reason:
            for instance in instance_rows:
                same_identity = (
                    str(instance["cell_id"]) == str(source_row["cell_id"])
                    and str(instance["pert_id"]) == str(source_row["pert_id"])
                    and str(instance["pert_type"]) == str(source_row["pert_type"])
                    and _canonical_physical_unit(instance["pert_dose_unit"])
                    == _canonical_physical_unit(source_row["pert_dose_unit"])
                    == str(qc["canonical_dose_unit"])
                    and _canonical_physical_unit(instance["pert_time_unit"])
                    == _canonical_physical_unit(source_row["pert_time_unit"])
                    == str(qc["canonical_time_unit"])
                )
                same_values = np.isclose(float(instance["pert_dose"]), float(source_row["pert_dose"])) and np.isclose(
                    float(instance["pert_time"]), float(source_row["pert_time"])
                )
                if not same_identity or not same_values:
                    reason = "source_instance_semantics_mismatch"
                    break
        plates = tuple(sorted({str(row["rna_plate"]) for row in instance_rows}))
        control_key_suffix = (
            _normalized_number(source_row["pert_time"]),
            str(source_row["pert_time_unit"]),
        )
        control_ids = tuple(sorted({
            control_id
            for plate in plates
            for control_id in controls_by_plate_time.get(
                (str(source_row["cell_id"]), plate, *control_key_suffix), ()
            )
        }))
        if not reason and (
            not plates
            or any(
                (str(source_row["cell_id"]), plate, *control_key_suffix)
                not in controls_by_plate_time
                for plate in plates
            )
        ):
            reason = "time_matched_vehicle_control_unavailable_on_contributing_plate"
        if not reason and sig_id not in column_index:
            reason = "signature_missing_from_level5_matrix"
        pert_row = pert_by_id.loc[str(source_row["pert_id"])] if str(source_row["pert_id"]) in pert_by_id.index else None
        if not reason and (pert_row is None or isinstance(pert_row, pd.DataFrame)):
            reason = "compound_metadata_missing_or_duplicated"

        if reason:
            quarantine_rows.append({
                "sig_id": sig_id,
                "cell_id": str(source_row["cell_id"]),
                "pert_id": str(source_row["pert_id"]),
                "pert_type": str(source_row["pert_type"]),
                "pert_dose": str(source_row["pert_dose"]),
                "pert_dose_unit": str(source_row["pert_dose_unit"]),
                "pert_time": str(source_row["pert_time"]),
                "pert_time_unit": str(source_row["pert_time_unit"]),
                "quarantine_reason": reason,
            })
            continue

        dose = float(source_row["pert_dose"])
        duration = float(source_row["pert_time"])
        condition_payload = {
            "cell_id": str(source_row["cell_id"]),
            "pert_id": str(source_row["pert_id"]),
            "dose_value": _normalized_number(dose),
            "dose_unit": str(qc["canonical_dose_unit"]),
            "time_value": _normalized_number(duration),
            "time_unit": str(qc["canonical_time_unit"]),
        }
        biological_unit_uid = _uid("clue_condition", condition_payload)
        control_group_uid = _uid("clue_control_group", {
            "cell_id": str(source_row["cell_id"]), "control_instance_ids": control_ids,
        })
        replicate_group_uid = _uid("clue_replicate_group", {"source_instance_ids": distil_ids})
        compound_uid = f"clue:{source_row['pert_id']}"
        action_uid = _uid("clue_action", condition_payload)
        assert isinstance(pert_row, pd.Series)
        qualified_rows.append({
            "observation_uid": f"clue_signature:{sig_id}",
            "source_uid": str(candidate["source_id"]),
            "publication_uid": str(candidate["publication_uid"]),
            "accession_uid": str(candidate["source_accession"]),
            "parent_accession_uid": str(candidate["parent_accession"]),
            "biological_unit_uid": biological_unit_uid,
            "subject_or_donor_uid": "explicit_unavailable",
            "sample_uid": f"clue_signature:{sig_id}",
            "cell_or_tissue_context": f"clue_cell_model:{source_row['cell_id']}",
            "cell_id": str(source_row["cell_id"]),
            "cns_context_class": str(candidate["cns_context_class"]),
            "compound_uid": compound_uid,
            "compound_name": str(source_row["pert_iname"]),
            "pubchem_cid_or_explicit_unavailable": str(pert_row["pubchem_cid"]) if str(pert_row["pubchem_cid"]) not in {"", "-666"} else "explicit_unavailable",
            "canonical_smiles_or_explicit_unavailable": str(pert_row["canonical_smiles"]) if str(pert_row["canonical_smiles"]) not in {"", "-666"} else "explicit_unavailable",
            "dose_value": dose,
            "dose_unit": str(qc["canonical_dose_unit"]),
            "raw_dose_value": str(source_row["pert_dose"]),
            "raw_dose_unit": str(source_row["pert_dose_unit"]),
            "time_value": duration,
            "time_unit": str(qc["canonical_time_unit"]),
            "raw_time_value": str(source_row["pert_time"]),
            "raw_time_unit": str(source_row["pert_time_unit"]),
            "assay_uid": "clue_assay:L1000",
            "modality": "transcriptomics",
            "feature_namespace": "NCBI_Entrez_Gene",
            "value_unit": "Level5_COMPZ_MODZ_consensus_z_score",
            "qc_grade": str(qc["grade"]),
            "action_uid": action_uid,
            "replicate_group_uid": replicate_group_uid,
            "source_instance_count": len(distil_ids),
            "source_instance_ids": "|".join(distil_ids),
            "rna_plate_ids": "|".join(plates),
            "control_group_uid": control_group_uid,
            "control_instance_ids": "|".join(control_ids),
            "gctx_column_index": int(column_index[sig_id]),
            "source_sig_id": sig_id,
        })

    candidates = pd.DataFrame(qualified_rows)
    if candidates.empty:
        raise ValueError("No NPC/NEU compound conditions satisfy perturbation M1")
    candidates = candidates.sort_values(
        ["biological_unit_uid", "source_instance_count", "source_sig_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    duplicate_mask = candidates.duplicated("biological_unit_uid", keep="first")
    duplicates = candidates[duplicate_mask]
    for row in duplicates.to_dict(orient="records"):
        quarantine_rows.append({
            "sig_id": row["source_sig_id"], "cell_id": row["cell_id"],
            "pert_id": row["compound_uid"], "pert_type": "trt_cp",
            "pert_dose": row["raw_dose_value"], "pert_dose_unit": row["raw_dose_unit"],
            "pert_time": row["raw_time_value"], "pert_time_unit": row["raw_time_unit"],
            "quarantine_reason": "duplicate_condition_signature_not_counted",
        })
    selected = candidates[~duplicate_mask].sort_values("gctx_column_index", kind="mergesort").reset_index(drop=True)
    selected["value_row_index"] = np.arange(len(selected), dtype=np.int64)
    quarantine = pd.DataFrame(quarantine_rows).sort_values(
        ["quarantine_reason", "cell_id", "sig_id"], kind="mergesort"
    ).reset_index(drop=True)

    actions = pd.DataFrame({
        "action_uid": selected["action_uid"],
        "intervention_type": "compound_perturbation",
        "agent_uid": selected["compound_uid"],
        "agent_set": selected["compound_uid"],
        "target_set_or_explicit_unknown": "explicit_unknown",
        "target_set": "explicit_unknown",
        "physical_dose": selected["dose_value"],
        "dose_value": selected["dose_value"],
        "dose_unit": selected["dose_unit"],
        "duration": selected["time_value"],
        "duration_unit": selected["time_unit"],
        "schedule": "explicit_unavailable",
        "route_or_not_applicable": "not_applicable_in_vitro",
        "route": "not_applicable_in_vitro",
        "cell_or_tissue_context": selected["cell_or_tissue_context"],
        "assay_uid": selected["assay_uid"],
        "assay_context": selected["cell_or_tissue_context"] + "|" + selected["assay_uid"],
        "state_scope": selected["cns_context_class"],
        "publication_uid": selected["publication_uid"],
        "source_publication": selected["publication_uid"],
        "accession_uid": selected["accession_uid"],
        "source_accession": selected["accession_uid"],
        "exposure_program_ref_or_explicit_unavailable": "explicit_unavailable",
        "exposure_program_ref": "explicit_unavailable",
        "occupancy_program_ref_or_explicit_unavailable": "explicit_unavailable",
        "occupancy_program_ref": "explicit_unavailable",
        "observation_uid": selected["observation_uid"],
        "control_group_uid": selected["control_group_uid"],
        "cns_context_class": selected["cns_context_class"],
        "qc_grade": selected["qc_grade"],
    })
    if actions["action_uid"].duplicated().any() or len(actions) != len(selected):
        raise ValueError("Structured action linkage is not one-to-one with selected conditions")
    if actions.astype(str).apply(lambda column: column.str.strip().eq("").any()).any():
        raise ValueError("Structured action linkage contains blank values")

    shsy5y = sig[sig["cell_id"].eq("SHSY5Y")]
    context_counts = selected["cell_id"].value_counts().sort_index().to_dict()
    source_context = source_subset["cell_id"].value_counts().sort_index().to_dict()
    diagnostics = {
        "source_signature_count": int(len(sig)),
        "source_subset_signature_count": int(len(source_subset)),
        "physical_dose_time_signature_count": int(source_subset["physical_axes_valid"].sum()),
        "replicate_control_valid_signature_count": int(len(candidates)),
        "duplicate_condition_signature_count": int(duplicate_mask.sum()),
        "selected_condition_count": int(len(selected)),
        "selected_context_counts": {key: int(value) for key, value in context_counts.items()},
        "source_context_counts": {key: int(value) for key, value in source_context.items()},
        "unique_compound_count": int(selected["compound_uid"].nunique()),
        "source_instance_count": int(selected["source_instance_count"].sum()),
        "unique_source_instance_count": int(len({value for group in selected["source_instance_ids"] for value in str(group).split("|")})),
        "unique_control_instance_count": int(len({value for group in selected["control_instance_ids"] for value in str(group).split("|")})),
        "unique_control_group_count": int(selected["control_group_uid"].nunique()),
        "quarantine_count": int(len(quarantine)),
        "quarantine_reason_counts": {key: int(value) for key, value in quarantine["quarantine_reason"].value_counts().sort_index().to_dict().items()},
        "shsy5y_signature_count": int(len(shsy5y)),
        "shsy5y_compound_signature_count": int(shsy5y["pert_type"].eq("trt_cp").sum()),
        "cns_applicability_evidence": {
            "NPC": {
                "cell_type": str(npc["cell_type"]),
                "sample_type": str(npc["sample_type"]),
                "primary_site": str(npc["primary_site"]),
                "modification": str(npc["modification"]),
            },
            "NEU": {
                "cell_type": str(neu["cell_type"]),
                "sample_type": str(neu["sample_type"]),
                "primary_site": str(neu["primary_site"]),
                "modification": str(neu["modification"]),
            },
            "classification_basis": "official TouchstoneV1.1 cell_info fields, not cell-name inference",
        },
    }
    return selected, actions, quarantine, diagnostics


def _semantic_axes(candidate: Mapping[str, object]) -> dict[str, dict[str, str]]:
    return {
        "study": {"status": "observed", "value": str(candidate["parent_accession"])},
        "biological_unit": {"status": "mapped", "value": "unique compound-dose-time condition within NPC/NEU context"},
        "species": {"status": "mapped", "value": "NCBITaxon:9606"},
        "disease_state": {"status": "not_applicable", "reason": "reference CNS-derived perturbation models are not a disease cohort"},
        "tissue": {"status": "mapped", "value": "CNS-derived in-vitro cellular model"},
        "brain_region": {"status": "not_applicable", "reason": "NPC and NEU models do not represent a sampled anatomical brain region"},
        "cell_type": {"status": "observed", "value": "NPC|NEU"},
        "assay": {"status": "mapped", "value": "clue_assay:L1000"},
        "modality": {"status": "observed", "value": "transcriptomics"},
        "feature": {"status": "mapped", "value": "NCBI_Entrez_Gene"},
        "time": {"status": "observed", "value": "pert_time canonicalized to h"},
        "dose": {"status": "observed", "value": "sig_info microM and inst_info um aliases canonicalized to uM"},
        "unit": {"status": "mapped", "value": "UCUM-compatible uM|h; Level5 consensus z-score value semantics"},
        "provenance": {"status": "observed", "value": "GSE92742 TouchstoneV1.1 Level5 plus sig/inst/cell/pert/gene metadata"},
    }


def build(repo_root: Path = REPO_ROOT) -> tuple[dict[str, object], dict[str, object]]:
    if repo_root.resolve() != REPO_ROOT:
        raise ValueError("Perturbation materialization is identity-bound to the repository root")
    config = load_config(REPO_ROOT / "Libs/configs/config_stack.yaml")
    policy = config["public_foundation_corpus"]["phase1e_perturbation_materialization"]
    candidate = policy["candidate"]
    qc = policy["qc"]
    assert isinstance(candidate, Mapping) and isinstance(qc, Mapping)

    release = _validate_release_evidence(policy)
    raw_path = _path(str(candidate["raw_value_path"]))
    archive_path = _path(str(candidate["raw_archive_path"]))
    raw_hash = sha256_file(raw_path)
    archive_hash = sha256_file(archive_path)
    if raw_path.stat().st_size != int(candidate["expected_raw_bytes"]) or raw_hash != candidate["expected_raw_sha256"]:
        raise ValueError("Touchstone GCTX bytes or SHA-256 do not match the frozen release contract")
    if archive_path.stat().st_size != int(candidate["expected_archive_bytes"]) or archive_hash != candidate["expected_archive_sha256"]:
        raise ValueError("Touchstone official archive bytes or SHA-256 do not match the frozen release contract")

    shape, dtype, column_ids, row_ids = _matrix_layout(raw_path)
    features = _feature_table(_path(str(candidate["metadata_paths"][-1])), row_ids)
    selected, actions, quarantine, diagnostics = _selection_tables(policy, column_ids)
    expected_diagnostics = {
        "source_subset_signature_count": int(qc["expected_source_subset_signature_count"]),
        "replicate_control_valid_signature_count": int(qc["expected_replicate_control_valid_signature_count"]),
        "duplicate_condition_signature_count": int(qc["expected_duplicate_condition_signature_count"]),
        "selected_condition_count": int(qc["expected_selected_condition_count"]),
        "selected_context_counts": dict(qc["expected_selected_context_counts"]),
        "quarantine_count": int(qc["expected_quarantine_count"]),
        "quarantine_reason_counts": dict(qc["expected_quarantine_reason_counts"]),
        "shsy5y_signature_count": int(qc["expected_shsy5y_signature_count"]),
        "shsy5y_compound_signature_count": int(qc["expected_shsy5y_compound_signature_count"]),
    }
    observed_diagnostics = {key: diagnostics.get(key) for key in expected_diagnostics}
    if observed_diagnostics != expected_diagnostics:
        raise ValueError(
            "Touchstone accession-scoped selection counts drifted: "
            + json.dumps({"expected": expected_diagnostics, "observed": observed_diagnostics}, sort_keys=True)
        )
    if len(selected) < int(qc["minimum_selected_condition_count"]):
        raise ValueError("Qualified perturbation conditions fall below the locked QC threshold")
    if len(features) != int(qc["expected_feature_count"]) or shape[1] != len(features):
        raise ValueError("Touchstone feature axis fails the locked full-feature contract")

    output_root = _path(str(policy["output_root"]))
    staging_root = output_root / ".phase1e_clue_perturbation_staging"
    ensure_directory(staging_root)
    with tempfile.TemporaryDirectory(prefix="clue-npc-neu-", dir=staging_root) as temporary:
        stage = Path(temporary)
        store = stage / "expression.zarr"
        initialize_zarr_store(store, {
            "format": "zarr_v2_uncompressed",
            "value_semantics": policy["parser"]["value_semantics"],
            "source_matrix": _relative(raw_path),
            "row_axis": "observations.parquet:value_row_index",
            "column_axis": "features.parquet:feature_index",
        })
        selected_indices = selected["gctx_column_index"].to_numpy(dtype=np.int64)
        matrix_stats: dict[str, object] = {
            "matrix_shape": [int(len(selected)), int(len(features))],
            "source_matrix_shape": [int(shape[0]), int(shape[1])],
            "source_dtype": str(dtype),
            "stored_dtype": "float32",
            "finite_values": True,
            "nonzero_value_count": 0,
            "minimum_value": None,
            "maximum_value": None,
            "scalar_value_count": int(len(selected) * len(features)),
        }

        with h5py.File(raw_path, "r") as handle:
            matrix = handle[MATRIX_DATASET]

            def loader(slices: tuple[slice, ...]) -> np.ndarray:
                row_slice, feature_slice = slices
                source_rows = selected_indices[row_slice]
                chunk = np.asarray(matrix[source_rows, feature_slice], dtype=np.float32)
                if not np.isfinite(chunk).all():
                    raise ValueError("Touchstone selected Level5 values contain NaN or Inf")
                matrix_stats["nonzero_value_count"] = int(matrix_stats["nonzero_value_count"]) + int(np.count_nonzero(chunk))
                chunk_min = float(chunk.min())
                chunk_max = float(chunk.max())
                current_min = matrix_stats["minimum_value"]
                current_max = matrix_stats["maximum_value"]
                matrix_stats["minimum_value"] = chunk_min if current_min is None else min(float(current_min), chunk_min)
                matrix_stats["maximum_value"] = chunk_max if current_max is None else max(float(current_max), chunk_max)
                return chunk

            write_zarr_array_from_loader(
                store,
                "values",
                (len(selected), len(features)),
                (int(policy["parser"]["row_chunk_size"]), len(features)),
                np.dtype("float32"),
                loader,
                attrs={"feature_namespace": "NCBI_Entrez_Gene", "unit": policy["parser"]["value_semantics"]},
            )

        write_parquet_atomic(selected, stage / "observations.parquet")
        write_parquet_atomic(features, stage / "features.parquet")
        write_parquet_atomic(actions, stage / "actions.parquet")
        write_parquet_atomic(quarantine, stage / "quarantine.parquet")
        write_json(stage / "matrix_manifest.json", matrix_stats)
        output_names = (
            "expression.zarr", "observations.parquet", "features.parquet",
            "actions.parquet", "quarantine.parquet", "matrix_manifest.json",
        )
        records = [_record(stage / name, name) for name in output_names]
        block_sha256 = _content_hash(records)
        destination = output_root / "blocks" / str(candidate["source_id"]) / str(candidate["source_subset_id"]) / block_sha256[:16]

        raw_binding = _binding(raw_path, known_sha256=raw_hash)
        archive_binding = _binding(archive_path, known_sha256=archive_hash)
        input_paths = [
            *[_path(str(value)) for value in candidate["metadata_paths"]],
            *[_path(str(value)) for value in candidate["release_evidence_paths"]],
        ]
        input_bindings = [raw_binding, archive_binding, *[_binding(path) for path in input_paths]]
        manifest = {
            "schema_version": MANIFEST_VERSION,
            "producer": BUILDER_ID,
            "created_at": _now(),
            "canonical_uid": CANONICAL_UID,
            "content_address": f"sha256:{block_sha256}",
            "block_sha256": block_sha256,
            "family_id": candidate["family_id"],
            "source_id": candidate["source_id"],
            "source_accession": candidate["source_accession"],
            "parent_accession": candidate["parent_accession"],
            "source_subset_id": candidate["source_subset_id"],
            "publication_accession_group": candidate["publication_accession_group"],
            "mirror_group": candidate["mirror_group"],
            "cns_source": True,
            "cns_context_class": candidate["cns_context_class"],
            "role": "qualified_for_mainline",
            "admission_allowed": True,
            "value_block_format": "zarr",
            "value_block_path": "expression.zarr",
            "sample_metadata_path": "observations.parquet",
            "feature_metadata_path": "features.parquet",
            "action_catalog_path": "actions.parquet",
            "quarantine_path": "quarantine.parquet",
            "matrix_manifest_path": "matrix_manifest.json",
            "observation_count": int(len(selected)),
            "biological_unit_count": int(selected["biological_unit_uid"].nunique()),
            "effective_sample_size": float(selected["biological_unit_uid"].nunique()),
            "effective_sample_size_formula": policy["biological_unit_policy"]["effective_sample_size_formula"],
            "biological_unit_policy": policy["biological_unit_policy"],
            "action_count": int(len(actions)),
            "action_linkage_complete": True,
            "real_value_matrix": True,
            "compound_dose_time_axes_complete": True,
            "observation_value_join_contract": {
                "observation_axis": "observations.parquet:value_row_index",
                "feature_axis": "features.parquet:feature_index|feature_id",
                "value_axis": "expression.zarr:values[value_row_index,feature_index]",
                "feature_namespace": "NCBI_Entrez_Gene",
                "value_unit": "Level5_COMPZ_MODZ_consensus_z_score",
                "long_form_values_materialized": False,
                "long_form_omission_reason": f"matrix-native storage avoids duplicating {len(selected) * len(features):,} observation-feature values",
            },
            "qc_grade": qc["grade"],
            "qc_grade_rationale": qc["grade_rationale"],
            "qc": {**diagnostics, **matrix_stats, "feature_count": int(len(features)), "status": "passed"},
            "mapping_confidence": "official_source_ids_exact_with_physical_unit_mapping_and_instance_control_lineage",
            "mapping_version": policy["parser"]["mapping_version"],
            "unit_mapping_version": policy["parser"]["unit_mapping_version"],
            "ontology_versions": {
                "species": "NCBITaxon:9606", "compound": "CLUE_pert_id_TouchstoneV1.1",
                "feature": "NCBI_Entrez_Gene_GSE92742", "assay": "CLUE_L1000",
                "unit": "UCUM_uM_h", "cell_context": "CLUE_cell_info_TouchstoneV1.1",
            },
            "semantic_axes": _semantic_axes(candidate),
            "semantic_axes_complete": True,
            "split_role": "unassigned_pending_nested_split_freeze",
            "source_release": {
                "official_url": candidate["official_value_url"],
                "release_version": candidate["source_release_version"],
                "retrieval_date": candidate["retrieval_date"],
                "publication_uid": candidate["publication_uid"],
                "training_evaluation_role": "internal_mainline_training_and_prespecified_evaluation_within_registered_nonprofit_research_group_no_asset_redistribution",
                "access_class": "registered_nonprofit_research_group_only",
                "license_role": policy["license_role"],
            },
            "parser": policy["parser"],
            "input_bindings": input_bindings,
            "block_files": records,
            "blocker_codes": [
                "second_independent_disease_source_missing",
                "second_independent_atlas_source_missing",
                "second_independent_perturbation_source_missing",
                "nested_split_freeze_pending",
                "real_domain_leakage_audit_pending",
            ],
        }
        write_json(stage / "block_manifest.json", manifest)
        if destination.exists():
            existing = _load_json(destination / "block_manifest.json")
            if existing.get("block_sha256") != block_sha256:
                raise ValueError(f"Content-address collision at {destination}")
            manifest = existing
        else:
            ensure_directory(destination.parent)
            stage.replace(destination)

    manifest_path = destination / "block_manifest.json"
    block_manifest_audit_path = _path(str(policy["block_manifest_audit_path"]))
    write_json(block_manifest_audit_path, manifest)
    delta_row = {
        "delta_version": "phase1e-source-qualification-delta-v1",
        "entry_id": "phase1e_clue_npc_neu_perturbation_mainline",
        "source_id": candidate["source_id"],
        "source_accession": candidate["source_accession"],
        "prior_decision": "qualified_for_weak_calibration",
        "decision": "qualified_for_mainline",
        "admission_allowed": "true",
        "scope": "source_accession_only",
        "frozen_ledger_path": str(policy["source_ledger_path"]),
        "frozen_ledger_sha256": sha256_file(_path(str(policy["source_ledger_path"]))),
        "block_manifest_path": _relative(manifest_path),
        "block_manifest_sha256": sha256_file(manifest_path),
        "training_evaluation_role": "internal_mainline_training_and_prespecified_evaluation_within_registered_nonprofit_research_group_no_asset_redistribution",
        "access_class": "registered_nonprofit_research_group_only",
        "status": "accepted",
    }
    upsert_accession_requalification_delta(_path(str(policy["qualification_delta_path"])), delta_row)

    common = {
        "schema_version": AUDIT_VERSION,
        "producer": BUILDER_ID,
        "created_at": _now(),
        "candidate_uid": "perturbation_clue_touchstone_v1_1_npc_neu",
        "source_id": candidate["source_id"],
        "source_accession": candidate["source_accession"],
        "canonical_uid": CANONICAL_UID,
        "canonical_block_manifest": _binding(manifest_path),
    }
    candidate_audit = {
        **common,
        "status": "passed_candidate_selected",
        "real_value_matrix": True,
        "source_matrix_shape": list(shape),
        "source_matrix_dtype": str(dtype),
        "candidate_context_class": candidate["cns_context_class"],
        "eligible_cell_ids": candidate["eligible_cell_ids"],
        "diagnostics": diagnostics,
        "license_release_evidence": release,
        "blocker_decomposition": [
            {
                "candidate_uid": "touchstone_npc_neu_trt_cp",
                "current_role": "qualified_for_mainline_accession_subset",
                "admission_allowed": True,
                "blocking_requirements": [],
                "estimated_recoverability": "high",
                "terminal_recommendation": "materialized",
            },
            {
                "candidate_uid": "touchstone_shsy5y",
                "current_role": "weak_calibration_only",
                "admission_allowed": False,
                "blocking_requirements": [
                    {
                        "requirement_id": "physical_compound_dose_axis",
                        "evidence_missing": "SHSY5Y has zero trt_cp signatures in TouchstoneV1.1",
                        "remediation": "do not use this context for compound-dose-time M1",
                    }
                ],
                "estimated_recoverability": "non_recoverable",
                "terminal_recommendation": "formally_reject_for_this_block",
            },
            {
                "candidate_uid": "touchstone_all_other_contexts",
                "current_role": "weak_calibration_only",
                "admission_allowed": False,
                "blocking_requirements": [
                    {
                        "requirement_id": "preregistered_cns_context",
                        "evidence_missing": "contexts are outside the locked NPC/NEU CNS-derived subset",
                        "remediation": "retain under source-wide weak-calibration role",
                    }
                ],
                "estimated_recoverability": "low",
                "terminal_recommendation": "defer",
            },
        ],
    }
    qc_audit = {
        **common,
        "status": "passed",
        "qc_grade": qc["grade"],
        "qc_grade_rationale": qc["grade_rationale"],
        "thresholds": qc,
        "results": {**diagnostics, **manifest["qc"]},
    }
    ess_audit = {
        **common,
        "status": "passed",
        "effective_sample_size": manifest["effective_sample_size"],
        "ess_unit": "unique_cell_context_compound_dose_time_condition",
        "formula": policy["biological_unit_policy"]["effective_sample_size_formula"],
        "raw_valid_signature_count": diagnostics["replicate_control_valid_signature_count"],
        "duplicate_condition_signature_count": diagnostics["duplicate_condition_signature_count"],
        "shared_controls_counted_in_ess": False,
        "source_instances_counted_individually_in_ess": False,
        "technical_replicates_counted_individually_in_ess": False,
        "cell_line_or_donor_inflation_allowed": False,
    }
    actions_path = destination / "actions.parquet"
    action_linkage = {
        **common,
        "status": "passed",
        "action_linkage_complete": True,
        "qualified_action_count": int(len(actions)),
        "linked_observation_count": int(len(selected)),
        "one_to_one_observation_action_linkage": True,
        "action_catalog": _binding(actions_path),
        "required_fields": list(actions.columns),
        "blank_field_count": 0,
        "explicit_unknown_target_count": int(len(actions)),
        "explicit_unavailable_exposure_count": int(len(actions)),
        "explicit_unavailable_occupancy_count": int(len(actions)),
        "explicit_unavailable_values_are_not_bootstrap_priors": True,
    }
    for path_key, payload in (
        ("candidate_audit_path", candidate_audit),
        ("qc_audit_path", qc_audit),
        ("ess_audit_path", ess_audit),
        ("action_linkage_path", action_linkage),
    ):
        write_json(_path(str(policy[path_key])), payload)

    admission = {
        **common,
        "status": "accepted_first_qualified_perturbation_block",
        "candidate_role_before": "qualified_for_weak_calibration",
        "candidate_role_after": "qualified_for_mainline",
        "source_family_role_unchanged": "qualified_for_weak_calibration",
        "admission_allowed": True,
        "scope": "source_accession_only",
        "qc_grade": manifest["qc_grade"],
        "effective_sample_size": manifest["effective_sample_size"],
        "real_value_matrix": manifest["real_value_matrix"],
        "compound_dose_time_axes_complete": manifest["compound_dose_time_axes_complete"],
        "action_linkage_complete": manifest["action_linkage_complete"],
        "source_qualification_delta": _binding(_path(str(policy["qualification_delta_path"]))),
        "block_manifest_audit": _binding(block_manifest_audit_path),
        "candidate_audit": _binding(_path(str(policy["candidate_audit_path"]))),
        "qc_audit": _binding(_path(str(policy["qc_audit_path"]))),
        "ess_audit": _binding(_path(str(policy["ess_audit_path"]))),
        "action_linkage": _binding(_path(str(policy["action_linkage_path"]))),
        "license_role": policy["license_role"],
        "remaining_g2_blockers": manifest["blocker_codes"],
        "role_transition_justification": "The accession-scoped NPC/NEU trt_cp subset has full Level-5 values, physical dose/time, source-instance and control lineage, condition-level ESS, and complete actions under CLUE internal nonprofit research restrictions; all other CLUE contexts remain weak-only.",
    }
    write_json(_path(str(policy["admission_audit_path"])), admission)
    return manifest, admission


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize the CLUE NPC/NEU perturbation M1 block.")
    parser.parse_args()
    manifest, admission = build(REPO_ROOT)
    print(json.dumps({
        "status": admission["status"],
        "canonical_uid": manifest["canonical_uid"],
        "block_sha256": manifest["block_sha256"],
        "observation_count": manifest["observation_count"],
        "effective_sample_size": manifest["effective_sample_size"],
        "action_count": manifest["action_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
