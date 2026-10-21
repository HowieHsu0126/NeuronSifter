from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path
import sys

import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, resolve_repo_path, write_tsv
from Libs.utils.data.screening_pipeline import (
    canonical_gene_entity_table,
    canonical_smiles_key,
    compound_descriptor_rows,
    evidence_score_from_affinity,
    first_nonempty,
    murcko_scaffold,
    normalize_symbol,
    normalize_text,
    parse_numeric,
    stable_uid,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the screening drug-target-ready registry from public pharmacology sources.")
    parser.add_argument(
        "--config",
        default="Libs/configs/config_stack.yaml",
        help="YAML configuration file for the raw and processed registries.",
    )
    return parser.parse_args()


def screening_root(config: dict, section_name: str) -> Path:
    return resolve_repo_path(config["screening_pipeline"][section_name]["local_root"])


def disease_patterns(config: dict) -> list[str]:
    return [normalize_text(value) for value in config["screening_pipeline"]["drug_target_ready_registry"]["disease_name_patterns"]]


def matches_disease_context(name: str, patterns: list[str]) -> bool:
    normalized_name = normalize_text(name)
    return any(pattern in normalized_name for pattern in patterns)


def open_targets_disease_ids(config: dict) -> tuple[set[str], dict[str, str]]:
    base = resolve_repo_path("Input/raw/open_targets/downloads/releases/25.03/disease")
    disease_frame = pd.concat([pd.read_parquet(path, columns=["id", "name"]) for path in sorted(base.glob("*.parquet"))], ignore_index=True)
    patterns = disease_patterns(config)
    filtered = disease_frame.loc[disease_frame["name"].astype(str).apply(lambda value: matches_disease_context(value, patterns))].copy()
    return set(filtered["id"].astype(str).tolist()), dict(zip(filtered["id"].astype(str), filtered["name"].astype(str), strict=False))


def open_targets_target_metadata(target_ids: set[str]) -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    base = resolve_repo_path("Input/raw/open_targets/downloads/releases/25.03/target")
    metadata_by_target: dict[str, dict[str, object]] = {}
    uniprot_to_symbol: dict[str, str] = {}
    columns = ["id", "approvedSymbol", "approvedName", "targetClass", "tractability", "proteinIds"]
    for path in sorted(base.glob("part-*.parquet")):
        frame = pd.read_parquet(path, columns=columns)
        frame = frame.loc[frame["id"].astype(str).isin(target_ids)].copy()
        if frame.empty:
            continue
        for row in frame.itertuples(index=False):
            target_id = str(row.id)
            target_class_payload = row.targetClass if isinstance(row.targetClass, (list, tuple, np.ndarray)) else (row.targetClass or [])
            tractability_payload = row.tractability if isinstance(row.tractability, (list, tuple, np.ndarray)) else (row.tractability or [])
            protein_id_payload = row.proteinIds if isinstance(row.proteinIds, (list, tuple, np.ndarray)) else (row.proteinIds or [])
            metadata_by_target[target_id] = {
                "approved_symbol": str(row.approvedSymbol),
                "approved_name": str(row.approvedName),
                "target_class": first_nonempty(
                    item.get("label", "")
                    for item in target_class_payload
                ),
                "tractability_bucket": tractability_bucket(tractability_payload),
            }
            for protein_record in protein_id_payload:
                protein_id = str(protein_record.get("id", "")).strip()
                if protein_id:
                    uniprot_to_symbol.setdefault(protein_id, str(row.approvedSymbol))
    return metadata_by_target, uniprot_to_symbol


def tractability_bucket(tractability_payload: object) -> str:
    if tractability_payload is None:
        return ""
    records = list(tractability_payload)
    approved_drug = any(item.get("id") == "Approved Drug" and bool(item.get("value")) for item in records)
    if approved_drug:
        return "approved_drug"
    high_quality_ligand = any(item.get("id") in {"High-Quality Ligand", "Structure with Ligand"} and bool(item.get("value")) for item in records)
    if high_quality_ligand:
        return "small_molecule_supported"
    antibody_supported = any(item.get("modality") == "AB" and bool(item.get("value")) for item in records)
    if antibody_supported:
        return "antibody_supported"
    return "unknown"


def open_targets_target_evidence(config: dict, disease_ids: set[str]) -> dict[str, dict[str, object]]:
    base = resolve_repo_path("Input/raw/open_targets/downloads/releases/25.03/association_overall_direct")
    evidence_by_target: dict[str, dict[str, object]] = defaultdict(lambda: {"score": 0.0, "count": 0, "public_url": "https://platform.opentargets.org/"})
    for path in sorted(base.glob("part-*.parquet")):
        frame = pd.read_parquet(path, columns=["diseaseId", "targetId", "score", "evidenceCount"])
        frame = frame.loc[frame["diseaseId"].astype(str).isin(disease_ids)].copy()
        if frame.empty:
            continue
        grouped = frame.groupby("targetId", as_index=False).agg({"score": "max", "evidenceCount": "sum"})
        for row in grouped.itertuples(index=False):
            target_id = str(row.targetId)
            evidence_by_target[target_id]["score"] = max(float(row.score), float(evidence_by_target[target_id]["score"]))
            evidence_by_target[target_id]["count"] += int(row.evidenceCount)
    return evidence_by_target


def open_targets_known_drugs(config: dict, disease_ids: set[str]) -> tuple[list[dict[str, object]], set[str]]:
    base = resolve_repo_path("Input/raw/open_targets/downloads/releases/25.03/known_drug")
    rows: list[dict[str, object]] = []
    compound_ids: set[str] = set()
    columns = ["drugId", "targetId", "diseaseId", "prefName", "drugType", "mechanismOfAction", "approvedSymbol", "targetName"]
    for path in sorted(base.glob("part-*.parquet")):
        frame = pd.read_parquet(path, columns=columns)
        frame = frame.loc[frame["diseaseId"].astype(str).isin(disease_ids)].copy()
        if frame.empty:
            continue
        for row in frame.itertuples(index=False):
            compound_id = str(row.drugId)
            compound_ids.add(compound_id)
            rows.append(
                {
                    "source_repository": "open_targets",
                    "compound_key": compound_id,
                    "target_id": str(row.targetId),
                    "target_symbol": str(row.approvedSymbol),
                    "target_name": str(row.targetName),
                    "drug_name": str(row.prefName),
                    "canonical_smiles": "",
                    "inchi_key": "",
                    "relation_type": "known_drug",
                    "affinity_type": "",
                    "affinity_value": None,
                    "affinity_unit": "",
                    "action_type": str(row.mechanismOfAction),
                    "evidence_score": 1.0,
                    "disease_context": str(row.diseaseId),
                    "public_url": "https://platform.opentargets.org/",
                    "chembl_id": compound_id if compound_id.startswith("CHEMBL") else "",
                    "drugcentral_struct_id": "",
                    "bindingdb_ligand_id": "",
                }
            )
    return rows, compound_ids


def open_targets_compounds(compound_ids: set[str]) -> dict[str, dict[str, str]]:
    base = resolve_repo_path("Input/raw/open_targets/downloads/releases/25.03/drug_molecule")
    compound_by_id: dict[str, dict[str, str]] = {}
    columns = ["id", "canonicalSmiles", "inchiKey", "name", "drugType"]
    for path in sorted(base.glob("part-*.parquet")):
        frame = pd.read_parquet(path, columns=columns)
        frame = frame.loc[frame["id"].astype(str).isin(compound_ids)].copy()
        if frame.empty:
            continue
        for row in frame.itertuples(index=False):
            compound_by_id[str(row.id)] = {
                "preferred_name": str(row.name),
                "canonical_smiles": str(row.canonicalSmiles),
                "inchi_key": str(row.inchiKey),
                "drug_type": str(row.drugType),
            }
    return compound_by_id


def drugcentral_edges() -> tuple[list[dict[str, object]], dict[str, dict[str, str]]]:
    structures_path = resolve_repo_path("Input/raw/drugcentral/downloads/releases/2021_09_01/structures.smiles.tsv")
    interaction_path = resolve_repo_path("Input/raw/drugcentral/downloads/releases/2021_09_01/drug.target.interaction.tsv.gz")
    structure_lookup: dict[str, dict[str, str]] = {}
    with open(structures_path, encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            struct_id = str(row["ID"])
            structure_lookup[struct_id] = {
                "preferred_name": str(row["INN"]),
                "canonical_smiles": str(row["SMILES"]),
                "inchi_key": str(row["InChIKey"]),
            }

    rows: list[dict[str, object]] = []
    with gzip.open(interaction_path, "rt", encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if str(row["ORGANISM"]).strip() != "Homo sapiens":
                continue
            affinity_value = parse_numeric(first_nonempty([row["ACT_VALUE"]]))
            rows.append(
                {
                    "source_repository": "drugcentral",
                    "compound_key": str(row["STRUCT_ID"]),
                    "target_symbol": str(row["GENE"]),
                    "target_name": str(row["TARGET_NAME"]),
                    "drug_name": str(row["DRUG_NAME"]),
                    "canonical_smiles": structure_lookup.get(str(row["STRUCT_ID"]), {}).get("canonical_smiles", ""),
                    "inchi_key": structure_lookup.get(str(row["STRUCT_ID"]), {}).get("inchi_key", ""),
                    "relation_type": "drug_target_interaction",
                    "affinity_type": str(row["ACT_TYPE"]),
                    "affinity_value": affinity_value,
                    "affinity_unit": str(row["ACT_UNIT"]),
                    "action_type": str(row["ACTION_TYPE"]),
                    "evidence_score": max(0.1, evidence_score_from_affinity(affinity_value)),
                    "disease_context": "Alzheimer disease|normal aging",
                    "public_url": first_nonempty([row["ACT_SOURCE_URL"], row["MOA_SOURCE_URL"], "https://drugcentral.org/download"]),
                    "chembl_id": "",
                    "drugcentral_struct_id": str(row["STRUCT_ID"]),
                    "bindingdb_ligand_id": "",
                }
            )
    return rows, structure_lookup


def bindingdb_edges(uniprot_to_symbol: dict[str, str]) -> list[dict[str, object]]:
    zip_path = resolve_repo_path("Input/raw/bindingdb/downloads/releases/BindingDB_All_202604_tsv.zip")
    rows: list[dict[str, object]] = []
    with zipfile.ZipFile(zip_path) as archive:
        member_name = archive.namelist()[0]
        with archive.open(member_name) as handle:
            reader = csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8", errors="ignore"), delimiter="\t")
            for row in reader:
                if "human" not in normalize_text(row["Target Source Organism According to Curator or DataSource"]):
                    continue
                target_symbol = ""
                for chain_index in range(1, 51):
                    uniprot_id = str(row.get(f"UniProt (SwissProt) Primary ID of Target Chain {chain_index}", "")).strip()
                    if uniprot_id and uniprot_id in uniprot_to_symbol:
                        target_symbol = uniprot_to_symbol[uniprot_id]
                        break
                if not target_symbol:
                    continue
                affinity_pairs = [
                    ("Ki", parse_numeric(row["Ki (nM)"])),
                    ("Kd", parse_numeric(row["Kd (nM)"])),
                    ("IC50", parse_numeric(row["IC50 (nM)"])),
                ]
                affinity_type = ""
                affinity_value = None
                for current_type, current_value in affinity_pairs:
                    if current_value is not None:
                        affinity_type = current_type
                        affinity_value = current_value
                        break
                if affinity_value is None:
                    continue
                chembl_id = str(row.get("ChEMBL ID of Ligand", "")).strip()
                bindingdb_ligand_id = str(row.get("BindingDB MonomerID", "")).strip()
                rows.append(
                    {
                        "source_repository": "bindingdb",
                        "compound_key": chembl_id or bindingdb_ligand_id,
                        "target_symbol": target_symbol,
                        "target_name": str(row["Target Name"]),
                        "drug_name": str(row["BindingDB Ligand Name"]),
                        "canonical_smiles": str(row["Ligand SMILES"]),
                        "inchi_key": str(row["Ligand InChI Key"]),
                        "relation_type": "measured_affinity",
                        "affinity_type": affinity_type,
                        "affinity_value": affinity_value,
                        "affinity_unit": "nM",
                        "action_type": "",
                        "evidence_score": evidence_score_from_affinity(affinity_value),
                        "disease_context": "Alzheimer disease|normal aging",
                        "public_url": first_nonempty([row["Link to Ligand-Target Pair in BindingDB"], "https://www.bindingdb.org/rwd/bind/index.jsp"]),
                        "chembl_id": chembl_id,
                        "drugcentral_struct_id": "",
                        "bindingdb_ligand_id": bindingdb_ligand_id,
                    }
                )
    return rows


def chembl_compound_lookup(compound_ids: set[str]) -> dict[str, dict[str, str]]:
    path = resolve_repo_path("Input/raw/chembl/downloads/releases/chembl_36_chemreps.txt.gz")
    lookup: dict[str, dict[str, str]] = {}
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            chembl_id = str(row["chembl_id"])
            if chembl_id not in compound_ids:
                continue
            lookup[chembl_id] = {
                "canonical_smiles": str(row["canonical_smiles"]),
                "inchi_key": str(row["standard_inchi_key"]),
            }
    return lookup


def target_mapping(
    target_symbol: str,
    target_name: str,
    gene_lookup: dict[str, dict[str, object]],
) -> tuple[str, str, str]:
    normalized_symbol = normalize_symbol(target_symbol)
    if normalized_symbol in gene_lookup:
        record = gene_lookup[normalized_symbol]
        return str(record["canonical_entity_id"]), str(record["canonical_entity_label"]), "exact_canonical_entity_label"
    normalized_name = normalize_symbol(target_name)
    if normalized_name in gene_lookup:
        record = gene_lookup[normalized_name]
        return str(record["canonical_entity_id"]), str(record["canonical_entity_label"]), "exact_source_gene_symbol"
    return "", "", ""


def merge_compounds(
    candidate_rows: list[dict[str, object]],
    open_targets_compound_lookup: dict[str, dict[str, str]],
    drugcentral_structures: dict[str, dict[str, str]],
    chembl_lookup: dict[str, dict[str, str]],
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    compound_state: dict[str, dict[str, object]] = {}
    compound_uid_by_key: dict[str, str] = {}
    inchi_to_uid: dict[str, str] = {}
    smiles_to_uid: dict[str, str] = {}
    exclusions: list[dict[str, object]] = []

    for row in candidate_rows:
        chembl_id = str(row["chembl_id"]).strip()
        drugcentral_struct_id = str(row["drugcentral_struct_id"]).strip()
        bindingdb_ligand_id = str(row["bindingdb_ligand_id"]).strip()
        canonical_id = chembl_id or (f"drugcentral:{drugcentral_struct_id}" if drugcentral_struct_id else f"bindingdb:{bindingdb_ligand_id}")
        namespace = "chembl_id" if chembl_id else ("drugcentral_struct_id" if drugcentral_struct_id else "bindingdb_ligand_id")

        open_targets_record = open_targets_compound_lookup.get(chembl_id, {})
        drugcentral_record = drugcentral_structures.get(drugcentral_struct_id, {})
        chembl_record = chembl_lookup.get(chembl_id, {})
        canonical_smiles = first_nonempty(
            [
                row["canonical_smiles"],
                open_targets_record.get("canonical_smiles", ""),
                drugcentral_record.get("canonical_smiles", ""),
                chembl_record.get("canonical_smiles", ""),
            ]
        )
        canonical_smiles = canonical_smiles_key(canonical_smiles) or canonical_smiles
        inchi_key = first_nonempty(
            [
                row["inchi_key"],
                open_targets_record.get("inchi_key", ""),
                drugcentral_record.get("inchi_key", ""),
                chembl_record.get("inchi_key", ""),
            ]
        )
        preferred_name = first_nonempty(
            [
                row["drug_name"],
                open_targets_record.get("preferred_name", ""),
                drugcentral_record.get("preferred_name", ""),
            ]
        )

        resolved_compound_uid = ""
        if chembl_id and chembl_id in compound_uid_by_key:
            resolved_compound_uid = compound_uid_by_key[chembl_id]
        elif inchi_key and inchi_key in inchi_to_uid:
            resolved_compound_uid = inchi_to_uid[inchi_key]
        elif canonical_smiles and canonical_smiles in smiles_to_uid:
            resolved_compound_uid = smiles_to_uid[canonical_smiles]
        else:
            resolved_compound_uid = stable_uid("compound", canonical_id or preferred_name or row["source_repository"])

        if resolved_compound_uid not in compound_state:
            compound_state[resolved_compound_uid] = {
                "compound_uid": resolved_compound_uid,
                "canonical_compound_id": canonical_id,
                "canonical_id_namespace": namespace,
                "preferred_name": preferred_name,
                "canonical_smiles": canonical_smiles,
                "inchi_key": inchi_key,
                "murcko_scaffold": murcko_scaffold(canonical_smiles),
                "source_names": set(),
                "chembl_id": chembl_id,
                "drugcentral_struct_id": drugcentral_struct_id,
                "bindingdb_ligand_id": bindingdb_ligand_id,
            }
        state = compound_state[resolved_compound_uid]
        state["source_names"].add(str(row["source_repository"]))
        if chembl_id:
            compound_uid_by_key[chembl_id] = resolved_compound_uid
        if drugcentral_struct_id:
            compound_uid_by_key[f"drugcentral:{drugcentral_struct_id}"] = resolved_compound_uid
        if bindingdb_ligand_id:
            compound_uid_by_key[f"bindingdb:{bindingdb_ligand_id}"] = resolved_compound_uid
        if inchi_key:
            inchi_to_uid[inchi_key] = resolved_compound_uid
        if canonical_smiles:
            smiles_to_uid[canonical_smiles] = resolved_compound_uid
        if state["canonical_compound_id"] != canonical_id and canonical_id:
            exclusions.append(
                {
                    "exclusion_uid": stable_uid("compound_conflict", resolved_compound_uid, canonical_id),
                    "exclusion_scope": "compound_canonicalization",
                    "source_repository": str(row["source_repository"]),
                    "source_record_id": canonical_id,
                    "exclusion_reason": "compound_identity_conflict",
                    "note": f"Resolved into existing compound_uid {resolved_compound_uid}.",
                }
            )

    normalized_compounds = {
        compound_uid: {
            **state,
            "source_count": len(state["source_names"]),
            "source_names": "|".join(sorted(state["source_names"])),
            "note": "",
        }
        for compound_uid, state in compound_state.items()
    }
    return normalized_compounds, exclusions


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    output_root = screening_root(config, "drug_target_ready_registry")
    ensure_directory(output_root)
    relative_paths = config["screening_pipeline"]["drug_target_ready_registry"]["relative_paths"]

    _, gene_lookup, duplicate_labels = canonical_gene_entity_table(config)
    disease_ids, disease_name_lookup = open_targets_disease_ids(config)
    open_targets_evidence = open_targets_target_evidence(config, disease_ids)
    known_drug_rows, known_drug_compounds = open_targets_known_drugs(config, disease_ids)
    open_targets_target_ids = set(open_targets_evidence) | {str(row["target_id"]) for row in known_drug_rows}
    target_metadata, uniprot_to_symbol = open_targets_target_metadata(open_targets_target_ids)
    open_targets_compounds_lookup = open_targets_compounds(known_drug_compounds)
    drugcentral_rows, drugcentral_structures = drugcentral_edges()
    bindingdb_rows = bindingdb_edges(uniprot_to_symbol)
    all_candidate_rows = known_drug_rows + drugcentral_rows + bindingdb_rows
    pipeline_logger.progress(
        "loaded_drug_target_inputs",
        open_targets_target_count=len(open_targets_evidence),
        open_targets_known_drug_count=len(known_drug_rows),
        drugcentral_edge_count=len(drugcentral_rows),
        bindingdb_edge_count=len(bindingdb_rows),
        candidate_edge_count=len(all_candidate_rows),
    )
    chembl_ids = {str(row["chembl_id"]).strip() for row in all_candidate_rows if str(row["chembl_id"]).strip()}
    chembl_lookup = chembl_compound_lookup(chembl_ids)
    compound_state, compound_exclusions = merge_compounds(
        all_candidate_rows,
        open_targets_compounds_lookup,
        drugcentral_structures,
        chembl_lookup,
    )

    target_rows: dict[str, dict[str, object]] = {}
    target_evidence_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    exclusion_rows: list[dict[str, object]] = []
    exclusion_rows.extend(compound_exclusions)
    for duplicate_label in sorted(duplicate_labels):
        exclusion_rows.append(
            {
                "exclusion_uid": stable_uid("duplicate_gene_label", duplicate_label),
                "exclusion_scope": "target_mapping",
                "source_repository": "alignment_ready",
                "source_record_id": duplicate_label,
                "exclusion_reason": "duplicate_canonical_entity_label",
                "note": "Retained only the highest-source_count mapping target.",
            }
        )

    compound_uid_lookup = {compound_uid: compound_uid for compound_uid in compound_state}
    compound_uid_by_chembl = {state["chembl_id"]: compound_uid for compound_uid, state in compound_state.items() if state["chembl_id"]}
    compound_uid_by_dc = {state["drugcentral_struct_id"]: compound_uid for compound_uid, state in compound_state.items() if state["drugcentral_struct_id"]}
    compound_uid_by_bindingdb = {state["bindingdb_ligand_id"]: compound_uid for compound_uid, state in compound_state.items() if state["bindingdb_ligand_id"]}

    for target_id, evidence in sorted(open_targets_evidence.items()):
        metadata = target_metadata.get(target_id)
        if metadata is None:
            continue
        canonical_entity_id, canonical_entity_label, mapping_method = target_mapping(
            metadata["approved_symbol"],
            metadata["approved_name"],
            gene_lookup,
        )
        if not canonical_entity_id:
            exclusion_rows.append(
                {
                    "exclusion_uid": stable_uid("ot_target_unmapped", target_id),
                    "exclusion_scope": "target_mapping",
                    "source_repository": "open_targets",
                    "source_record_id": target_id,
                    "exclusion_reason": "unmapped_target_symbol",
                    "note": metadata["approved_symbol"],
                }
            )
            continue
        target_uid = stable_uid("target", canonical_entity_id)
        target_rows[target_uid] = {
            "target_uid": target_uid,
            "canonical_entity_id": canonical_entity_id,
            "canonical_entity_label": canonical_entity_label,
            "target_symbol": metadata["approved_symbol"],
            "target_name": metadata["approved_name"],
            "target_class": metadata["target_class"],
            "tractability_bucket": metadata["tractability_bucket"],
            "source_repository": "open_targets",
            "mapping_method": mapping_method,
            "mapping_confidence": "exact",
            "source_count": 1,
            "public_url": "https://platform.opentargets.org/",
            "note": "",
        }
        target_evidence_rows.append(
            {
                "evidence_uid": stable_uid("ot_evidence", target_uid),
                "target_uid": target_uid,
                "source_repository": "open_targets",
                "disease_context": "|".join(sorted({disease_name_lookup[disease_id] for disease_id in disease_ids if disease_id in disease_name_lookup})),
                "evidence_score": float(evidence["score"]),
                "evidence_count": int(evidence["count"]),
                "public_url": str(evidence["public_url"]),
                "note": "disease_target_association",
            }
        )

    for row in all_candidate_rows:
        canonical_entity_id, canonical_entity_label, mapping_method = target_mapping(row["target_symbol"], row["target_name"], gene_lookup)
        if not canonical_entity_id:
            exclusion_rows.append(
                {
                    "exclusion_uid": stable_uid("edge_unmapped_target", row["source_repository"], row["target_symbol"], row["compound_key"]),
                    "exclusion_scope": "drug_target_edge",
                    "source_repository": row["source_repository"],
                    "source_record_id": f"{row['compound_key']}::{row['target_symbol']}",
                    "exclusion_reason": "unmapped_target_symbol",
                    "note": row["target_name"],
                }
            )
            continue
        target_uid = stable_uid("target", canonical_entity_id)
        if target_uid not in target_rows:
            target_rows[target_uid] = {
                "target_uid": target_uid,
                "canonical_entity_id": canonical_entity_id,
                "canonical_entity_label": canonical_entity_label,
                "target_symbol": row["target_symbol"],
                "target_name": row["target_name"],
                "target_class": "",
                "tractability_bucket": "",
                "source_repository": row["source_repository"],
                "mapping_method": mapping_method,
                "mapping_confidence": "exact",
                "source_count": 1,
                "public_url": row["public_url"],
                "note": "",
            }

        compound_uid = ""
        if row["chembl_id"] and row["chembl_id"] in compound_uid_by_chembl:
            compound_uid = compound_uid_by_chembl[row["chembl_id"]]
        elif row["drugcentral_struct_id"] and row["drugcentral_struct_id"] in compound_uid_by_dc:
            compound_uid = compound_uid_by_dc[row["drugcentral_struct_id"]]
        elif row["bindingdb_ligand_id"] and row["bindingdb_ligand_id"] in compound_uid_by_bindingdb:
            compound_uid = compound_uid_by_bindingdb[row["bindingdb_ligand_id"]]
        if not compound_uid:
            exclusion_rows.append(
                {
                    "exclusion_uid": stable_uid("edge_unmapped_compound", row["source_repository"], row["compound_key"]),
                    "exclusion_scope": "drug_target_edge",
                    "source_repository": row["source_repository"],
                    "source_record_id": row["compound_key"],
                    "exclusion_reason": "unmapped_compound_identity",
                    "note": row["drug_name"],
                }
            )
            continue
        edge_rows.append(
            {
                "edge_uid": stable_uid("edge", compound_uid, target_uid, row["source_repository"], str(row["relation_type"])),
                "compound_uid": compound_uid,
                "target_uid": target_uid,
                "source_repository": row["source_repository"],
                "disease_context": row["disease_context"],
                "relation_type": row["relation_type"],
                "affinity_type": row["affinity_type"],
                "affinity_value": row["affinity_value"] if row["affinity_value"] is not None else "",
                "affinity_unit": row["affinity_unit"],
                "evidence_score": float(row["evidence_score"]),
                "action_type": row["action_type"],
                "public_url": row["public_url"],
                "note": row["drug_name"],
            }
        )

    compound_rows = []
    for compound_uid, state in sorted(compound_state.items()):
        compound_rows.append(
            {
                "compound_uid": compound_uid,
                "canonical_compound_id": state["canonical_compound_id"],
                "canonical_id_namespace": state["canonical_id_namespace"],
                "preferred_name": state["preferred_name"],
                "canonical_smiles": state["canonical_smiles"],
                "inchi_key": state["inchi_key"],
                "murcko_scaffold": state["murcko_scaffold"],
                "source_count": state["source_count"],
                "source_names": state["source_names"],
                "chembl_id": state["chembl_id"],
                "drugcentral_struct_id": state["drugcentral_struct_id"],
                "bindingdb_ligand_id": state["bindingdb_ligand_id"],
                "note": state["note"],
            }
        )
        feature_rows.extend(compound_descriptor_rows(compound_uid, state["canonical_smiles"], "rdkit"))

    exclusion_rows.append(
        {
            "exclusion_uid": stable_uid("dgidb_no_structured_export"),
            "exclusion_scope": "drug_target_edge",
            "source_repository": "dgidb",
            "source_record_id": "v.5.0.11",
            "exclusion_reason": "no_structured_public_interaction_export",
            "note": "Current DGIdb raw snapshot contains release archives but no stable public TSV/CSV interaction export.",
        }
    )

    output_paths = {name: output_root / relative_path for name, relative_path in relative_paths.items()}
    write_tsv(output_paths["target_catalog"], config["schemas"]["drug_target_target_fields"], sorted(target_rows.values(), key=lambda row: (row["canonical_entity_label"], row["target_uid"])))
    write_tsv(output_paths["compound_catalog"], config["schemas"]["drug_target_compound_fields"], compound_rows)
    write_tsv(output_paths["drug_target_edge_catalog"], config["schemas"]["drug_target_edge_fields"], sorted(edge_rows, key=lambda row: (row["compound_uid"], row["target_uid"], row["source_repository"])))
    write_tsv(output_paths["target_evidence_catalog"], config["schemas"]["drug_target_evidence_fields"], sorted(target_evidence_rows, key=lambda row: (row["target_uid"], row["source_repository"])))
    write_tsv(output_paths["compound_feature_catalog"], config["schemas"]["drug_target_compound_feature_fields"], sorted(feature_rows, key=lambda row: (row["compound_uid"], row["feature_name"])))
    write_tsv(output_paths["drug_target_exclusion_catalog"], config["schemas"]["drug_target_exclusion_fields"], sorted(exclusion_rows, key=lambda row: (row["exclusion_scope"], row["source_repository"], row["source_record_id"])))
    pipeline_logger.artifact(
        "drug_target_registry_written",
        artifact=output_root,
        target_row_count=len(target_rows),
        compound_row_count=len(compound_rows),
        edge_row_count=len(edge_rows),
        target_evidence_row_count=len(target_evidence_rows),
        feature_row_count=len(feature_rows),
        exclusion_row_count=len(exclusion_rows),
    )

    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=output_root,
        target_row_count=len(target_rows),
        compound_row_count=len(compound_rows),
        edge_row_count=len(edge_rows),
        target_evidence_row_count=len(target_evidence_rows),
        feature_row_count=len(feature_rows),
        exclusion_row_count=len(exclusion_rows),
    )

if __name__ == "__main__":
    main()
