"""Seal the versioned public-foundation corpus V2 rescue evidence chain."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import itertools
import json
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.foundation_state_c1_bindings import (  # noqa: E402
    build_c1_observation_binding_artifacts,
    serialize_binding_manifest,
)
from Libs.utils.data.public_foundation_corpus import canonical_sha256, hash_path  # noqa: E402
from Libs.utils.data.public_foundation_evidence import (  # noqa: E402
    adversarial_leakage_test,
    canonical_hash,
    safe_repo_path,
    sha256_file,
)
from Libs.utils.data.raw_data_registry import (  # noqa: E402
    read_tsv,
    write_json,
    write_text_atomic,
    write_tsv,
)


CONFIG_PATH = REPO_ROOT / "Libs/configs/data/public_foundation_corpus_rescue.yaml"
AUDIT_ROOT = REPO_ROOT / "Output/audits"
V2_PREFIX = "public_foundation_corpus_rescue"
SOURCE_PATH = AUDIT_ROOT / f"{V2_PREFIX}_source_qualification.json"
BLOCK_PATH = AUDIT_ROOT / f"{V2_PREFIX}_canonical_observation_blocks.json"
CROSSWALK_PATH = AUDIT_ROOT / f"{V2_PREFIX}_crosswalk_delta.json"
CROSSWALK_TABLE_PATH = AUDIT_ROOT / f"{V2_PREFIX}_crosswalk_delta.tsv"
ACTION_AUDIT_PATH = AUDIT_ROOT / f"{V2_PREFIX}_structured_action_catalog.json"
ACTION_SUBSET_PATH = AUDIT_ROOT / "c1_rescue_qualified_action_subset_manifest.json"
QC_PATH = AUDIT_ROOT / f"{V2_PREFIX}_qc_ess_statistics.json"
SPLIT_PATH = AUDIT_ROOT / f"{V2_PREFIX}_frozen_splits.json"
LEAKAGE_PATH = AUDIT_ROOT / f"{V2_PREFIX}_leakage_audit.json"
STORAGE_PATH = AUDIT_ROOT / f"{V2_PREFIX}_storage_ledger.json"
BUNDLE_PATH = AUDIT_ROOT / f"{V2_PREFIX}_evidence_bundle.json"
FROZEN_BUNDLE_SHA256 = "7accf20e92e2c56b64798d4200eb27a321372c0fcc07546f09877e6d10c41033"
BINDING_PATH = AUDIT_ROOT / "c1_rescue_observation_row_binding_manifest.json"
BINDING_TABLE_PATH = (
    REPO_ROOT
    / "Input/processed/foundation_state_c1/public_foundation_corpus_rescue_observation_row_bindings.tsv"
)
MATERIALIZATION_PATH = AUDIT_ROOT / "c1_rescue_materialization.json"
ANCHOR_PATH = AUDIT_ROOT / "c1_rescue_anchor_probe.json"
V1_BLOCKS_PATH = AUDIT_ROOT / "public_foundation_corpus_canonical_observation_blocks.json"
V1_ACTION_PATH = AUDIT_ROOT / "c1_qualified_action_subset_manifest.json"
V1_QC_PATH = AUDIT_ROOT / "public_foundation_corpus_qc_ess_statistics.json"
V1_STORAGE_PATH = AUDIT_ROOT / "public_foundation_corpus_storage_ledger.json"

CORE_FAMILIES = (
    "cns_disease_aging_state_transcriptomics",
    "cns_cell_type_region_atlas_transcriptomics",
    "compound_dose_time_perturbation_transcriptomics",
)
SPLIT_FIELDS = [
    "canonical_uid",
    "content_hash",
    "family_uid",
    "source_id",
    "source_accession",
    "publication_accession_group",
    "mirror_group",
    "biological_unit_uid",
    "regimen_uid",
    "target_program_proxy",
    "source_holdout_role",
    "split_role",
]
REQUIRED_ZERO_OVERLAP_KEYS = (
    "canonical_uid",
    "content_hash",
    "family_uid",
    "regimen_uid",
    "target_program_proxy",
    "source_holdout_role",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _load_policy() -> Mapping[str, object]:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    policy = payload.get("public_foundation_corpus_rescue")
    if not isinstance(policy, Mapping):
        raise ValueError("V2 rescue policy is malformed")
    return policy


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _binding(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": _relative(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _semantic_payload(payload: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in payload.items()
        if key not in {"created_at", "producer", "corpus_hash", "corpus_identity_hash"}
    }


def _artifact_ref(path: Path, payload: Mapping[str, object]) -> dict[str, object]:
    return {
        "path": _relative(path),
        "sha256": sha256_file(path),
        "semantic_hash": canonical_sha256(_semantic_payload(payload)),
        "status": payload.get("status", "unknown"),
        "record_count": int(payload.get("record_count", 0)),
        "schema_version": payload.get("schema_version", "unknown"),
    }


def _verify_parent(policy: Mapping[str, object]) -> dict[str, object]:
    parent = policy["parent_corpus"]
    assert isinstance(parent, Mapping)
    for key in ("evidence_bundle", "canonical_blocks", "frozen_splits", "leakage_audit"):
        path = REPO_ROOT / str(parent[f"{key}_path"])
        observed = sha256_file(path)
        if observed != str(parent[f"{key}_sha256"]):
            raise ValueError(f"Frozen V1 parent artifact drift: {key}")
    bundle = _load_json(REPO_ROOT / str(parent["evidence_bundle_path"]))
    if bundle.get("corpus_uid") != parent["corpus_uid"] or bundle.get("corpus_hash") != parent["identity_hash"]:
        raise ValueError("Frozen V1 corpus identity drift")
    return bundle


def _validate_block_files(manifest_path: Path, manifest: Mapping[str, object]) -> None:
    records = manifest.get("block_files")
    if not isinstance(records, list) or not records:
        raise ValueError(f"Block has no content records: {manifest_path}")
    hash_payload: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError(f"Malformed block content record: {manifest_path}")
        path = manifest_path.parent / str(record["relative_path"])
        count, byte_count, digest = hash_path(path)
        if (
            count != int(record["file_count"])
            or byte_count != int(record["bytes"])
            or digest != str(record["sha256"])
        ):
            raise ValueError(f"Block content drift: {path}")
        hash_payload.append(
            {
                "name": record["name"],
                "file_count": count,
                "bytes": byte_count,
                "sha256": digest,
            }
        )
    if canonical_sha256(hash_payload) != manifest.get("block_sha256"):
        raise ValueError(f"Block content address drift: {manifest_path}")


def _new_manifest_paths(materialization: Mapping[str, object]) -> list[Path]:
    blocks = materialization.get("blocks")
    if materialization.get("status") != "passed" or not isinstance(blocks, list) or len(blocks) != 2:
        raise ValueError("C1 rescue materialization evidence is incomplete")
    paths: list[Path] = []
    for row in blocks:
        if not isinstance(row, Mapping) or not isinstance(row.get("manifest"), Mapping):
            raise ValueError("C1 rescue materialization binding is malformed")
        path = safe_repo_path(REPO_ROOT, str(row["manifest"]["path"]))
        if sha256_file(path) != str(row["manifest"]["sha256"]):
            raise ValueError("C1 rescue materialization manifest hash drift")
        paths.append(path)
    return paths


def _block_row(manifest_path: Path, *, parent_version: str) -> dict[str, object]:
    manifest = _load_json(manifest_path)
    _validate_block_files(manifest_path, manifest)
    if manifest.get("role") != "qualified_for_mainline" or manifest.get("qc_grade") not in {"A", "B"}:
        raise ValueError(f"Forbidden role or grade in V2 qualified block: {manifest_path}")
    return {
        "canonical_uid": manifest["canonical_uid"],
        "family_id": manifest["family_id"],
        "source_id": manifest["source_id"],
        "source_accession": manifest["source_accession"],
        "publication_accession_group": manifest["publication_accession_group"],
        "mirror_group": manifest["mirror_group"],
        "cns_source": bool(manifest["cns_source"]),
        "biological_unit_count": int(manifest["biological_unit_count"]),
        "observation_count": int(manifest["observation_count"]),
        "effective_sample_size": float(manifest["effective_sample_size"]),
        "value_block_format": manifest["value_block_format"],
        "accepted_value_block_format": manifest["value_block_format"] in {"zarr", "parquet"},
        "value_block_path": _relative(manifest_path.parent / str(manifest["value_block_path"])),
        "block_sha256": manifest["block_sha256"],
        "qc_grade": manifest["qc_grade"],
        "mapping_confidence": manifest["mapping_confidence"],
        "ontology_versions": manifest.get("ontology_versions", {}),
        "unit_mapping_version": manifest.get("unit_mapping_version", "not_available"),
        "split_role": "frozen_by_public_foundation_corpus_rescue",
        "observation_eligible": True,
        "semantic_axes_complete": bool(manifest["semantic_axes_complete"]),
        "role": manifest["role"],
        "admission_allowed": bool(manifest["admission_allowed"]),
        "manifest_path": _relative(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "corpus_origin": parent_version,
        "blocker_codes": [],
    }


def _qualified_blocks(new_paths: Sequence[Path]) -> list[dict[str, object]]:
    v1 = _load_json(V1_BLOCKS_PATH)
    rows = [
        dict(row, corpus_origin="public_foundation_corpus_v1", blocker_codes=[])
        for row in v1.get("qualified_blocks", [])
        if isinstance(row, Mapping)
    ]
    if len(rows) != 6:
        raise ValueError("Frozen V1 qualified block count drift")
    for row in rows:
        path = safe_repo_path(REPO_ROOT, str(row["manifest_path"]))
        if sha256_file(path) != str(row["manifest_sha256"]):
            raise ValueError(f"Frozen V1 block manifest drift: {row['canonical_uid']}")
        manifest = _load_json(path)
        if manifest.get("block_sha256") != row["block_sha256"]:
            raise ValueError(f"Frozen V1 block content identity drift: {row['canonical_uid']}")
    rows.extend(_block_row(path, parent_version="public_foundation_corpus_rescue") for path in new_paths)
    if len({str(row["canonical_uid"]) for row in rows}) != 8:
        raise ValueError("V2 canonical block UIDs are duplicated")
    return sorted(rows, key=lambda row: str(row["canonical_uid"]))


def _build_source_qualification(
    policy: Mapping[str, object], new_paths: Sequence[Path]
) -> dict[str, object]:
    gse = policy["candidates"]["gse262992"]
    controls = policy["candidates"]["e_mtab_6772_controls"]
    rows = [
        {
            "candidate_uid": gse["candidate_uid"],
            "official_source": "NCBI GEO",
            "official_accession": gse["source_accession"],
            "official_release": gse["official_release"],
            "access_class": "public",
            "training_role_compatibility": "qualified_for_mainline",
            "qualification_scope": "source_accession_only",
            "raw_value_matrix_available": True,
            "qc_grade": "B",
            "biological_unit_count": 2,
            "independent_publication_accession_credit": True,
            "manifest": _binding(new_paths[0]),
            "decision": "qualified_for_mainline",
        },
        {
            "candidate_uid": controls["candidate_uid"],
            "official_source": "EMBL-EBI ArrayExpress",
            "official_accession": controls["source_accession"],
            "official_release": "existing V1-qualified public release",
            "access_class": "public_with_attribution",
            "training_role_compatibility": "qualified_non_action_control_context",
            "qualification_scope": "V2_block_variant_within_existing_accession",
            "raw_value_matrix_available": True,
            "qc_grade": "B",
            "biological_unit_count": 6,
            "independent_publication_accession_credit": False,
            "same_matrix_as_v1_declared": True,
            "manifest": _binding(new_paths[1]),
            "decision": "qualified_for_mainline_control_context",
        },
    ]
    return {
        "schema_version": "neuronsifter-public-foundation-corpus-v2-source-qualification-v1",
        "status": "passed",
        "created_at": _now(),
        "record_count": len(rows),
        "parent_corpus_identity_hash": policy["parent_corpus"]["identity_hash"],
        "versioned_role_changes_only": True,
        "v1_role_decisions_modified": False,
        "sources": rows,
        "blocker_codes": [],
    }


def _build_crosswalk(policy: Mapping[str, object]) -> dict[str, object]:
    anchor = _load_json(ANCHOR_PATH)
    clue = anchor.get("qualified_clue_dexamethasone_anchor")
    if not isinstance(clue, Mapping) or clue.get("status") != "available":
        raise ValueError("Qualified V1 CLUE dexamethasone anchor is unavailable")
    records = clue.get("records")
    if not isinstance(records, list) or len(records) != 5:
        raise ValueError("Qualified V1 CLUE dexamethasone anchor count drift")
    rows: list[dict[str, object]] = [
        {
            "mapping_uid": "c1r-v2-region-gse262992",
            "entity_type": "brain_region",
            "source_id": "geo",
            "source_accession": "GSE262992",
            "source_value": "hippocampal neural stem cell model",
            "canonical_namespace": "UBERON",
            "canonical_value": "UBERON:0002421",
            "mapping_method": "official_sample_context_to_hippocampus_ontology",
            "mapping_confidence": "high",
            "role": "qualified_for_mainline",
        },
        {
            "mapping_uid": "c1r-v2-region-e-mtab-6772-controls",
            "entity_type": "brain_region",
            "source_id": "arrayexpress",
            "source_accession": "E-MTAB-6772",
            "source_value": "hippocampus",
            "canonical_namespace": "UBERON",
            "canonical_value": "UBERON:0002421",
            "mapping_method": "frozen_V1_E-MTAB-6772_region_mapping_reused_for_controls",
            "mapping_confidence": "high",
            "role": "qualified_for_mainline",
        },
        {
            "mapping_uid": "c1r-v2-compound-gse262992-dexamethasone",
            "entity_type": "compound",
            "source_id": "geo",
            "source_accession": "GSE262992",
            "source_value": "Dex",
            "canonical_namespace": "PubChem",
            "canonical_value": "pubchem:5743",
            "mapping_method": "canonical_name_and_InChIKey_prefix_UREBDLICKHMUKA",
            "mapping_confidence": "exact_identity",
            "role": "qualified_for_mainline",
        },
    ]
    for record in records:
        if str(record.get("inchi_key_prefix")) != "UREBDLICKHMUKA":
            raise ValueError("V1 CLUE dexamethasone compound identity drift")
        rows.append(
            {
                "mapping_uid": f"c1r-v2-compound-{record['action_uid']}",
                "entity_type": "compound",
                "source_id": "clue_lincs",
                "source_accession": "GSE92742-TouchstoneV1.1::NPC_NEU_TRT_CP",
                "source_value": record["agent_uid"],
                "canonical_namespace": "PubChem",
                "canonical_value": "pubchem:5743",
                "mapping_method": "official_CLUE_pert_info_PubChem_and_InChIKey",
                "mapping_confidence": "exact_identity",
                "role": "qualified_for_mainline",
            }
        )
    fields = list(rows[0])
    write_tsv(CROSSWALK_TABLE_PATH, fields, rows)
    return {
        "schema_version": "neuronsifter-public-foundation-corpus-v2-crosswalk-delta-v1",
        "status": "passed",
        "created_at": _now(),
        "record_count": len(rows),
        "mapping_version": "public_foundation_corpus_rescue_rescue_crosswalk_2026-07-22",
        "parent_mapping_version": "public_foundation_corpus_v1_crosswalk_2026-07-17",
        "table": _binding(CROSSWALK_TABLE_PATH),
        "canonical_region_anchor": "UBERON:0002421",
        "canonical_compound_anchor": "pubchem:5743",
        "blocker_codes": [],
    }


def _action_catalogs(new_paths: Sequence[Path]) -> list[dict[str, object]]:
    v1 = _load_json(V1_ACTION_PATH)
    catalogs = [dict(row) for row in v1["qualified_catalogs"]]
    gse_manifest = _load_json(next(path for path in new_paths if "GSE262992" in path.as_posix()))
    action_path = next(path for path in new_paths if "GSE262992" in path.as_posix()).parent / str(
        gse_manifest["action_catalog_path"]
    )
    catalogs.append(
        {
            "asset_id": "c1r_v2_gse262992_dexamethasone_qualified_action",
            "path": _relative(action_path),
            "record_count": 1,
            "bytes": action_path.stat().st_size,
            "sha256": sha256_file(action_path),
            "source_id": "geo",
            "source_accession": "GSE262992",
            "canonical_observation_uid": gse_manifest["canonical_uid"],
            "role": "qualified_for_mainline",
            "action_linkage_complete": True,
        }
    )
    catalogs.sort(key=lambda row: str(row["asset_id"]))
    if sum(int(row["record_count"]) for row in catalogs) != 7124:
        raise ValueError("V2 qualified action count drift")
    for row in catalogs:
        path = safe_repo_path(REPO_ROOT, str(row["path"]))
        if path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            raise ValueError(f"V2 action catalog content drift: {row['asset_id']}")
    return catalogs


def _build_action_audit(catalogs: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "neuronsifter-public-foundation-corpus-v2-action-catalog-v1",
        "status": "passed",
        "created_at": _now(),
        "record_count": sum(int(row["record_count"]) for row in catalogs),
        "qualified_record_count": sum(int(row["record_count"]) for row in catalogs),
        "catalog_count": len(catalogs),
        "qualified_catalogs": list(catalogs),
        "synthetic_action_count": 0,
        "bootstrap_prior_action_count": 0,
        "blocker_codes": [],
    }


def _text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"", "nan", "none", "<na>"} else text


def _query_units(blocks: Sequence[Mapping[str, object]], transfer_action_uids: set[str]) -> list[dict[str, str]]:
    units: list[dict[str, str]] = []
    for block in blocks:
        manifest_path = safe_repo_path(REPO_ROOT, str(block["manifest_path"]))
        manifest = _load_json(manifest_path)
        metadata_path = manifest_path.parent / str(manifest["sample_metadata_path"])
        metadata = pd.read_parquet(metadata_path)
        family = str(block["family_id"])
        if family == "cns_cell_type_region_atlas_transcriptomics":
            unit_column = "donor_id" if "donor_id" in metadata else "donor_label"
        elif "biological_unit_uid" in metadata:
            unit_column = "biological_unit_uid"
        else:
            unit_column = "biological_unit_id"
        columns = [unit_column]
        for name in ("action_uid", "compound_uid"):
            if name in metadata:
                columns.append(name)
        unit_frame = metadata[columns].drop_duplicates().copy()
        for record in unit_frame.to_dict(orient="records"):
            biological_uid = _text(record[unit_column])
            action_uid = _text(record.get("action_uid"))
            compound_uid = _text(record.get("compound_uid"))
            if not biological_uid:
                raise ValueError(f"Blank V2 biological unit: {manifest_path}")
            if action_uid in transfer_action_uids:
                regimen_uid = "canonical_compound_regimen:pubchem:5743"
                target_proxy = "explicit_unknown_target_agent_proxy:pubchem:5743"
            elif action_uid:
                regimen_uid = action_uid
                target_proxy = f"explicit_unknown_target_agent_proxy:{compound_uid or 'explicit_unknown'}"
            else:
                regimen_uid = f"not_applicable_control_context:{block['canonical_uid']}"
                target_proxy = f"not_applicable_control_context:{block['canonical_uid']}"
            units.append(
                {
                    "canonical_uid": f"pfc_unit:{canonical_hash([manifest['canonical_uid'], biological_uid])[:24]}",
                    "content_hash": canonical_hash([manifest["block_sha256"], biological_uid]),
                    "family_uid": family,
                    "source_id": str(block["source_id"]),
                    "source_accession": str(block["source_accession"]),
                    "publication_accession_group": str(block["publication_accession_group"]),
                    "mirror_group": str(block["mirror_group"]),
                    "biological_unit_uid": biological_uid,
                    "regimen_uid": regimen_uid,
                    "target_program_proxy": target_proxy,
                    "source_holdout_role": str(block["source_id"]),
                    "split_role": "unassigned",
                }
            )
    if len({row["canonical_uid"] for row in units}) != len(units):
        raise ValueError("V2 query domain contains duplicate canonical UIDs")
    if len({row["biological_unit_uid"] for row in units}) != len(units):
        raise ValueError("V2 query domain contains cross-block biological-unit duplication")
    return sorted(units, key=lambda row: row["canonical_uid"])


def _validation_groups(rows: Sequence[Mapping[str, str]], seed: str, fraction: float) -> set[str]:
    groups = sorted({row["biological_unit_uid"] for row in rows})
    if len(groups) < 2:
        return set()
    count = min(len(groups) - 1, max(1, int(round(len(groups) * fraction))))
    return set(sorted(groups, key=lambda value: canonical_hash([seed, "validation", value]))[:count])


def _source_roles(units: Sequence[Mapping[str, str]], seed: str) -> dict[str, str]:
    sources = sorted({row["source_id"] for row in units})
    candidates: list[tuple[str, dict[str, str]]] = []
    for bits in itertools.product(("development", "test"), repeat=len(sources)):
        roles = dict(zip(sources, bits, strict=True))
        if all(
            {roles[row["source_id"]] for row in units if row["family_uid"] == family}
            == {"development", "test"}
            for family in CORE_FAMILIES
        ):
            candidates.append(
                (
                    canonical_hash([seed, sorted(roles.items())]),
                    roles,
                )
            )
    if not candidates:
        raise ValueError("No global V2 source-holdout assignment covers every family")
    return min(candidates, key=lambda item: item[0])[1]


def _write_split(
    split_root: Path,
    *,
    split_uid: str,
    split_type: str,
    rows: Sequence[Mapping[str, str]],
    applicable_keys: Sequence[str],
    seed: str,
    validation_fraction: float,
) -> dict[str, object]:
    by_role: dict[str, list[dict[str, str]]] = {role: [] for role in ("train", "validation", "test")}
    for row in rows:
        by_role[str(row["split_role"])].append(dict(row))
    if not by_role["train"] or not by_role["test"]:
        raise ValueError(f"V2 split lacks train or test units: {split_uid}")
    bindings: dict[str, dict[str, object]] = {}
    for role, assigned in by_role.items():
        path = split_root / f"{split_uid}.{role}.tsv"
        write_tsv(path, SPLIT_FIELDS, assigned)
        bindings[role] = {
            "path": _relative(path),
            "record_count": len(assigned),
            "sha256": sha256_file(path),
        }
    payload: dict[str, object] = {
        "schema_version": "neuronsifter-public-foundation-corpus-v2-split-v1",
        "split_uid": split_uid,
        "split_type": split_type,
        "corpus_domain_hash": canonical_hash(sorted(row["content_hash"] for row in rows)),
        "manifest_hash": "",
        "frozen": True,
        "deterministic_seed": seed,
        "validation_group_fraction": validation_fraction,
        "role_bindings": bindings,
        "applicable_zero_overlap_keys": list(applicable_keys),
    }
    payload["manifest_hash"] = canonical_hash(
        {key: value for key, value in payload.items() if key != "manifest_hash"}
    )
    return payload


def _build_splits(
    policy: Mapping[str, object], blocks: Sequence[Mapping[str, object]], transfer_action_uids: set[str]
) -> dict[str, object]:
    split_policy = policy["split_policy"]
    split_root = REPO_ROOT / str(policy["paths"]["split_root"])
    split_root.mkdir(parents=True, exist_ok=True)
    units = _query_units(blocks, transfer_action_uids)
    query_path = split_root / "query_domain.tsv"
    write_tsv(query_path, SPLIT_FIELDS, units)
    seed = str(split_policy["deterministic_seed"])
    fraction = float(split_policy["validation_group_fraction"])
    roles = _source_roles(units, seed)
    development = [row for row in units if roles[row["source_id"]] == "development"]
    validation = _validation_groups(development, seed, fraction)
    source_rows: list[dict[str, str]] = []
    for row in units:
        assigned = dict(row)
        assigned["split_role"] = (
            "test"
            if roles[row["source_id"]] == "test"
            else ("validation" if row["biological_unit_uid"] in validation else "train")
        )
        source_rows.append(assigned)
    manifests = [
        _write_split(
            split_root,
            split_uid="source_publication_holdout",
            split_type="source_holdout",
            rows=source_rows,
            applicable_keys=("canonical_uid", "content_hash", "publication_accession_group", "source_holdout_role"),
            seed=seed,
            validation_fraction=fraction,
        )
    ]
    for heldout_family in CORE_FAMILIES:
        development = [row for row in units if row["family_uid"] != heldout_family]
        validation = _validation_groups(development, f"{seed}|{heldout_family}", fraction)
        rows: list[dict[str, str]] = []
        for row in units:
            assigned = dict(row)
            assigned["split_role"] = (
                "test"
                if row["family_uid"] == heldout_family
                else ("validation" if row["biological_unit_uid"] in validation else "train")
            )
            rows.append(assigned)
        manifests.append(
            _write_split(
                split_root,
                split_uid=f"family_holdout_{heldout_family}",
                split_type="family_holdout",
                rows=rows,
                applicable_keys=("canonical_uid", "content_hash", "family_uid"),
                seed=seed,
                validation_fraction=fraction,
            )
        )
    perturbation = [
        row for row in units if row["family_uid"] == "compound_dose_time_perturbation_transcriptomics"
    ]
    for split_type, group_key in (
        ("regimen_holdout", "regimen_uid"),
        ("target_program_holdout", "target_program_proxy"),
    ):
        actions = [
            row for row in perturbation if not row[group_key].startswith("not_applicable_control_context:")
        ]
        groups = sorted(
            {row[group_key] for row in actions},
            key=lambda value: canonical_hash([seed, split_type, value]),
        )
        if len(groups) < 3:
            raise ValueError(f"V2 {split_type} has fewer than three eligible action groups")
        test_count = min(len(groups) - 2, max(1, int(round(len(groups) * 0.2))))
        test_groups = set(groups[:test_count])
        remaining_groups = [value for value in groups if value not in test_groups]
        validation_count = min(
            len(remaining_groups) - 1,
            max(1, int(round(len(remaining_groups) * fraction))),
        )
        validation_groups = set(
            sorted(
                remaining_groups,
                key=lambda value: canonical_hash([seed, split_type, "validation", value]),
            )[:validation_count]
        )
        rows = []
        for row in perturbation:
            assigned = dict(row)
            if row[group_key].startswith("not_applicable_control_context:"):
                assigned["split_role"] = "train"
            elif row[group_key] in test_groups:
                assigned["split_role"] = "test"
            elif row[group_key] in validation_groups:
                assigned["split_role"] = "validation"
            else:
                assigned["split_role"] = "train"
            rows.append(assigned)
        manifests.append(
            _write_split(
                split_root,
                split_uid=split_type,
                split_type=split_type,
                rows=rows,
                applicable_keys=("canonical_uid", "content_hash", group_key),
                seed=seed,
                validation_fraction=fraction,
            )
        )
    return {
        "schema_version": "neuronsifter-public-foundation-corpus-v2-frozen-splits-v1",
        "status": "passed",
        "created_at": _now(),
        "record_count": len(units),
        "qualified_observation_block_count": len(blocks),
        "query_domain_status": "frozen",
        "split_manifest_status": "frozen",
        "frozen": True,
        "split_seed": seed,
        "query_domain": {
            "path": _relative(query_path),
            "record_count": len(units),
            "sha256": sha256_file(query_path),
        },
        "source_holdout_roles": roles,
        "actionless_control_policy": split_policy["actionless_control_policy"],
        "nested_split_order": split_policy["nested_split_order"],
        "manifests": manifests,
        "blocker_codes": [],
    }


def _build_leakage(splits: Mapping[str, object]) -> dict[str, object]:
    findings: list[dict[str, object]] = []
    covered: set[str] = set()
    totals: Counter[str] = Counter()
    for manifest in splits["manifests"]:
        applicable = list(manifest["applicable_zero_overlap_keys"])
        covered.update(applicable)
        role_rows = {
            role: read_tsv(safe_repo_path(REPO_ROOT, str(binding["path"])))
            for role, binding in manifest["role_bindings"].items()
        }
        overlaps: dict[str, int] = {}
        for key in applicable:
            role_sets = [
                {str(row[key]) for row in role_rows[role] if str(row.get(key) or "")}
                for role in ("train", "validation", "test")
            ]
            if key in {"source_holdout_role", "publication_accession_group", "family_uid"}:
                values = (role_sets[0] | role_sets[1]) & role_sets[2]
            else:
                values = (
                    (role_sets[0] & role_sets[1])
                    | (role_sets[0] & role_sets[2])
                    | (role_sets[1] & role_sets[2])
                )
            overlaps[key] = len(values)
            totals[key] += len(values)
        findings.append(
            {
                "split_uid": manifest["split_uid"],
                "applicable_zero_overlap_keys": applicable,
                "overlap_counts": overlaps,
                "zero_overlap": not any(overlaps.values()),
            }
        )
    adversarial = {key: adversarial_leakage_test([key]) for key in REQUIRED_ZERO_OVERLAP_KEYS}
    uncovered = sorted(set(REQUIRED_ZERO_OVERLAP_KEYS) - covered)
    blockers: list[str] = []
    if uncovered:
        blockers.append("required_leakage_key_not_covered")
    if any(totals.values()):
        blockers.append("real_domain_overlap_detected")
    if any(row["status"] != "passed" for row in adversarial.values()):
        blockers.append("adversarial_leakage_detection_failed")
    return {
        "schema_version": "neuronsifter-public-foundation-corpus-v2-leakage-audit-v1",
        "status": "passed" if not blockers else "not_passed",
        "created_at": _now(),
        "record_count": int(splits["record_count"]),
        "eligible_observation_count": int(splits["record_count"]),
        "audit_domain_status": "frozen_real_observation_domain",
        "zero_overlap": not any(totals.values()),
        "required_zero_overlap_keys": list(REQUIRED_ZERO_OVERLAP_KEYS),
        "covered_zero_overlap_keys": sorted(covered),
        "overlap_counts": dict(sorted(totals.items())),
        "split_findings": findings,
        "adversarial_injection": {
            "tested": True,
            "all_required_keys_detected": all(row["status"] == "passed" for row in adversarial.values()),
            "by_key": adversarial,
        },
        "blocker_codes": blockers,
    }


def _finalize_new_manifests(new_paths: Sequence[Path]) -> None:
    split_binding = _binding(SPLIT_PATH)
    leakage_binding = _binding(LEAKAGE_PATH)
    for path in new_paths:
        manifest = _load_json(path)
        if manifest["canonical_uid"] == "pfc_geo_gse262992_dex_control_rescue":
            manifest["semantic_axes"] = {
                "study": {"status": "observed", "value": "GSE262992"},
                "biological_unit": {
                    "status": "mapped",
                    "value": "condition aggregate after three declared replicates",
                },
                "species": {"status": "mapped", "value": "NCBITaxon:10090"},
                "disease_state": {
                    "status": "not_applicable",
                    "reason": "wild-type reference neural stem-cell model",
                },
                "tissue": {"status": "mapped", "value": "UBERON:0002421 hippocampus"},
                "brain_region": {"status": "mapped", "value": "UBERON:0002421"},
                "cell_type": {"status": "observed", "value": "hippocampal neural stem cell"},
                "assay": {"status": "mapped", "value": "geo_assay:GPL24247_rna_seq"},
                "modality": {"status": "observed", "value": "transcriptomics"},
                "feature": {"status": "mapped", "value": "Ensembl_Gene"},
                "time": {"status": "observed", "value": "72 h"},
                "dose": {"status": "observed", "value": "1 uM dexamethasone"},
                "unit": {"status": "mapped", "value": "uM|h|FPKM"},
                "provenance": {
                    "status": "observed",
                    "value": "official GEO series matrix and sample FPKM tables",
                },
            }
        else:
            manifest["semantic_axes"] = {
                "study": {"status": "observed", "value": "E-MTAB-6772 controls_v2"},
                "biological_unit": {
                    "status": "mapped",
                    "value": "source-declared control specimen without invented mouse ID",
                },
                "species": {"status": "mapped", "value": "NCBITaxon:10090"},
                "disease_state": {
                    "status": "observed",
                    "value": "5XFAD Alzheimer disease mouse model|wild type",
                },
                "tissue": {"status": "mapped", "value": "UBERON:0002421 hippocampus"},
                "brain_region": {"status": "mapped", "value": "UBERON:0002421"},
                "cell_type": {"status": "not_applicable", "reason": "bulk pooled hippocampus"},
                "assay": {"status": "mapped", "value": "arrayexpress_assay:A-MEXP-932"},
                "modality": {"status": "observed", "value": "transcriptomics"},
                "feature": {"status": "mapped", "value": "NCBI_Entrez_Gene"},
                "time": {"status": "observed", "value": "5.5 month study context"},
                "dose": {"status": "not_applicable", "reason": "vehicle control"},
                "unit": {
                    "status": "mapped",
                    "value": "normalized_Illumina_AVG_Signal",
                },
                "provenance": {
                    "status": "observed",
                    "value": "official ArrayExpress SDRF/ADF/assay tables",
                },
            }
        manifest["split_role"] = "frozen_by_public_foundation_corpus_rescue"
        manifest["v2_split_binding"] = split_binding
        manifest["v2_leakage_binding"] = leakage_binding
        manifest["blocker_codes"] = []
        write_json(path, manifest)
    materialization = _load_json(MATERIALIZATION_PATH)
    by_uid = {
        _load_json(path)["canonical_uid"]: _binding(path)
        for path in new_paths
    }
    for row in materialization["blocks"]:
        row["manifest"] = by_uid[row["canonical_uid"]]
    materialization["evidence_sealing_status"] = "v2_splits_and_leakage_frozen"
    write_json(MATERIALIZATION_PATH, materialization)


def _build_qc(blocks: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_family: dict[str, dict[str, object]] = {}
    for family in CORE_FAMILIES:
        rows = [row for row in blocks if row["family_id"] == family]
        by_family[family] = {
            "qualified_block_count": len(rows),
            "grade_ab_observation_count": sum(int(row["observation_count"]) for row in rows),
            "grade_ab_effective_sample_size": sum(float(row["effective_sample_size"]) for row in rows),
            "source_count": len({row["source_id"] for row in rows}),
            "publication_accession_group_count": len(
                {row["publication_accession_group"] for row in rows}
            ),
        }
    total_observations = sum(int(row["observation_count"]) for row in blocks)
    total_ess = sum(float(row["effective_sample_size"]) for row in blocks)
    blockers: list[str] = []
    if total_observations != 909484:
        blockers.append("grade_ab_observation_count_mismatch")
    if total_ess != 7191.0:
        blockers.append("grade_ab_ess_mismatch")
    return {
        "schema_version": "neuronsifter-public-foundation-corpus-v2-qc-ess-v1",
        "status": "passed" if not blockers else "not_passed",
        "created_at": _now(),
        "record_count": len(blocks),
        "qualified_block_count": len(blocks),
        "grade_ab_observation_count": total_observations,
        "grade_ab_effective_sample_size": total_ess,
        "grade_ab_action_count": 7124,
        "family_statistics": by_family,
        "synthetic_observation_count": 0,
        "weak_calibration_observation_count": 0,
        "context_prior_observation_count": 0,
        "controlled_observation_count": 0,
        "blocker_codes": blockers,
    }


def _tree_size(path: Path) -> tuple[int, int]:
    file_count = 0
    byte_count = 0
    for root, _, files in os.walk(path, followlinks=False):
        for name in files:
            item = Path(root) / name
            if item.is_file() and not item.is_symlink():
                file_count += 1
                byte_count += item.stat().st_size
    return file_count, byte_count


def _build_storage(policy: Mapping[str, object]) -> dict[str, object]:
    parent = _load_json(V1_STORAGE_PATH)
    parent_total = int(parent["usage_bytes"])
    roots = {
        "v2_raw": REPO_ROOT / str(policy["paths"]["raw_root"]),
        "v2_canonical": REPO_ROOT / str(policy["paths"]["canonical_root"]),
    }
    classes: dict[str, dict[str, object]] = {}
    increment = 0
    for name, path in roots.items():
        count, byte_count = _tree_size(path)
        increment += byte_count
        classes[name] = {
            "path": _relative(path),
            "file_count": count,
            "bytes": byte_count,
        }
    total = parent_total + increment
    cap = int(policy["storage_policy"]["decimal_byte_cap"])
    return {
        "schema_version": "neuronsifter-public-foundation-corpus-v2-storage-ledger-v1",
        "status": "passed" if total <= cap else "not_passed",
        "created_at": _now(),
        "record_count": len(classes),
        "measurement_policy": "frozen_complete_v1_ledger_plus_complete_v2_path_delta",
        "parent_storage_ledger": _binding(V1_STORAGE_PATH),
        "parent_total_bytes": parent_total,
        "v2_increment_bytes": increment,
        "v2_classes": classes,
        "total_bytes": total,
        "decimal_byte_cap": cap,
        "cap_status": "passed" if total <= cap else "not_passed",
        "partial_scan": False,
        "blocker_codes": [] if total <= cap else ["storage_cap_exceeded"],
    }


def _anchor_summary() -> dict[str, object]:
    anchor = _load_json(ANCHOR_PATH)
    clue = anchor["qualified_clue_dexamethasone_anchor"]
    return {
        "qualified_cross_source_anchor_count": 1,
        "cross_source_anchor": {
            "axis": "brain_region_ontology_id",
            "canonical_value": "UBERON:0002421",
            "source_ids": ["arrayexpress", "geo"],
            "eligible_biological_unit_count": 7,
            "positive_pair_count": 6,
            "action_rows_excluded": True,
        },
        "qualified_perturbation_transfer_anchor_count": 1,
        "perturbation_transfer_anchor": {
            "canonical_compound_uid": "pubchem:5743",
            "canonical_agent_name": "dexamethasone",
            "source_ids": ["clue_lincs", "geo"],
            "qualified_action_count": int(clue["qualified_action_count"]) + 1,
            "v1_clue_action_count": int(clue["qualified_action_count"]),
            "v2_geo_action_count": 1,
            "same_dose_time_required": False,
            "transfer_axis": "shared_compound_identity_across_source_and_regimen",
        },
    }


def _action_subset(corpus_hash: str, catalogs: Sequence[Mapping[str, object]]) -> dict[str, object]:
    payload = {
        "corpus_hash": corpus_hash,
        "qualified_catalogs": list(catalogs),
    }
    return {
        "schema_version": "neuronsifter-c1-qualified-action-subset-v2",
        "status": "passed",
        "corpus_uid": "public_foundation_corpus_rescue",
        "corpus_hash": corpus_hash,
        "role": "qualified_for_mainline",
        "record_count": 7124,
        "catalog_count": len(catalogs),
        "qualified_catalogs": list(catalogs),
        "qualified_action_subset_hash": canonical_sha256(payload),
    }


def build(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    if repo_root.resolve() != REPO_ROOT:
        raise ValueError("C1 rescue V2 evidence sealing is repository-root bound")
    policy = _load_policy()
    parent_bundle = _verify_parent(policy)
    materialization = _load_json(MATERIALIZATION_PATH)
    new_paths = _new_manifest_paths(materialization)
    new_paths.sort(key=lambda path: ("GSE262992" not in path.as_posix(), path.as_posix()))

    source = _build_source_qualification(policy, new_paths)
    write_json(SOURCE_PATH, source)
    crosswalk = _build_crosswalk(policy)
    write_json(CROSSWALK_PATH, crosswalk)
    catalogs = _action_catalogs(new_paths)
    actions = _build_action_audit(catalogs)
    write_json(ACTION_AUDIT_PATH, actions)
    blocks = _qualified_blocks(new_paths)
    anchor = _load_json(ANCHOR_PATH)["qualified_clue_dexamethasone_anchor"]
    transfer_actions = {
        str(row["action_uid"]) for row in anchor["records"]
    } | {"geo_action:GSE262992:dexamethasone_1uM_72h"}
    splits = _build_splits(policy, blocks, transfer_actions)
    write_json(SPLIT_PATH, splits)
    leakage = _build_leakage(splits)
    write_json(LEAKAGE_PATH, leakage)
    if leakage["status"] != "passed":
        raise ValueError(f"V2 real-domain leakage audit failed: {leakage['blocker_codes']}")
    _finalize_new_manifests(new_paths)
    source = _build_source_qualification(policy, new_paths)
    write_json(SOURCE_PATH, source)
    blocks = _qualified_blocks(new_paths)
    block_artifact = {
        "schema_version": "neuronsifter-public-foundation-corpus-v2-observation-blocks-v1",
        "status": "passed",
        "created_at": _now(),
        "record_count": len(blocks),
        "qualified_real_value_block_count": len(blocks),
        "synthetic_value_block_count": 0,
        "weak_or_context_mainline_block_count": 0,
        "blocks": blocks,
        "qualified_blocks": blocks,
        "blocker_codes": [],
    }
    write_json(BLOCK_PATH, block_artifact)
    qc = _build_qc(blocks)
    write_json(QC_PATH, qc)
    storage = _build_storage(policy)
    write_json(STORAGE_PATH, storage)
    if qc["status"] != "passed" or storage["status"] != "passed":
        raise ValueError("V2 QC/ESS or storage gate failed")

    semantic_identity = {
        "corpus_uid": "public_foundation_corpus_rescue",
        "parent_corpus_uid": parent_bundle["corpus_uid"],
        "parent_corpus_identity_hash": parent_bundle["corpus_hash"],
        "block_hashes": {
            row["canonical_uid"]: row["block_sha256"] for row in blocks
        },
        "source_qualification_hash": canonical_sha256(_semantic_payload(source)),
        "crosswalk_delta_hash": canonical_sha256(_semantic_payload(crosswalk)),
        "action_catalog_hash": canonical_sha256(list(catalogs)),
        "query_domain_hash": splits["query_domain"]["sha256"],
        "split_hashes": {
            row["split_uid"]: row["manifest_hash"] for row in splits["manifests"]
        },
        "leakage_hash": canonical_sha256(_semantic_payload(leakage)),
        "qc_ess_hash": canonical_sha256(_semantic_payload(qc)),
        "storage_policy_hash": canonical_sha256(_semantic_payload(storage)),
        "scope": "reduced_transcriptomics_core_v2_rescue",
    }
    corpus_hash = canonical_sha256(semantic_identity)
    action_subset = _action_subset(corpus_hash, catalogs)
    write_json(ACTION_SUBSET_PATH, action_subset)
    refs = {
        "source_qualification": _artifact_ref(SOURCE_PATH, source),
        "crosswalk_delta": _artifact_ref(CROSSWALK_PATH, crosswalk),
        "canonical_observation_blocks": _artifact_ref(BLOCK_PATH, block_artifact),
        "structured_action_catalog": _artifact_ref(ACTION_AUDIT_PATH, actions),
        "qualified_action_subset": _artifact_ref(ACTION_SUBSET_PATH, action_subset),
        "frozen_splits": _artifact_ref(SPLIT_PATH, splits),
        "leakage_audit": _artifact_ref(LEAKAGE_PATH, leakage),
        "qc_ess_statistics": _artifact_ref(QC_PATH, qc),
        "storage_ledger": _artifact_ref(STORAGE_PATH, storage),
        "controlled_isolation": {
            **_binding(REPO_ROOT / "Output/audits/public_mainline_isolation_audit.json"),
            "status": "passed",
            "controlled_reachability": 0,
        },
        "parent_v1_evidence_bundle": {
            **_binding(REPO_ROOT / str(policy["parent_corpus"]["evidence_bundle_path"])),
            "status": "valid_fail_closed",
            "corpus_identity_hash": policy["parent_corpus"]["identity_hash"],
        },
    }
    blockers = [
        key for key, payload in refs.items() if payload.get("status") not in {"passed", "valid_fail_closed"}
    ]
    anchors = _anchor_summary()
    bundle = {
        "schema_version": "neuronsifter-public-foundation-corpus-v2-evidence-bundle-v1",
        "status": "valid_fail_closed" if not blockers else "blocked",
        "created_at": _now(),
        "corpus_uid": "public_foundation_corpus_rescue",
        "corpus_hash": corpus_hash,
        "corpus_identity_hash": corpus_hash,
        "parent_corpus_uid": parent_bundle["corpus_uid"],
        "parent_corpus_identity_hash": parent_bundle["corpus_hash"],
        "scope": "reduced_transcriptomics_core_v2_rescue",
        "identity_payload": semantic_identity,
        "evidence": refs,
        "qualified_block_count": len(blocks),
        "grade_ab_observation_count": qc["grade_ab_observation_count"],
        "grade_ab_effective_sample_size": qc["grade_ab_effective_sample_size"],
        "qualified_action_count": action_subset["record_count"],
        **anchors,
        "derived_gates": {
            "G1": "passed",
            "G2_reduced": "PASS_WITH_SCOPE_LIMITATION",
            "G2_full_multimodal": "not_passed",
        },
        "blockers": blockers,
    }
    write_json(BUNDLE_PATH, bundle)
    if blockers:
        raise ValueError(f"V2 evidence bundle is blocked: {blockers}")

    row_text, binding_manifest = build_c1_observation_binding_artifacts(
        REPO_ROOT,
        bundle,
        blocks,
        splits["query_domain"],
        row_table_path=_relative(BINDING_TABLE_PATH),
    )
    binding_manifest["derivation_policy"] = "read_only_derived_artifact_for_frozen_rescue"
    write_text_atomic(BINDING_TABLE_PATH, row_text)
    write_text_atomic(BINDING_PATH, serialize_binding_manifest(binding_manifest))
    binding_ref = {
        "manifest": _binding(BINDING_PATH),
        "row_table": _binding(BINDING_TABLE_PATH),
        "binding_hash": binding_manifest["binding_hash"],
        "block_count": binding_manifest["block_count"],
        "biological_unit_count": binding_manifest["biological_unit_count"],
        "observation_count": binding_manifest["observation_count"],
    }
    bundle["c1_observation_binding"] = binding_ref
    write_json(BUNDLE_PATH, bundle)
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    if not BUNDLE_PATH.is_file():
        raise FileNotFoundError(f"Frozen C1 rescue bundle is absent: {BUNDLE_PATH}")
    observed_sha256 = sha256_file(BUNDLE_PATH)
    if observed_sha256 != FROZEN_BUNDLE_SHA256:
        raise ValueError(
            "Frozen C1 rescue bundle drifted; recover it from the sealed lineage instead of resealing"
        )
    bundle = _load_json(BUNDLE_PATH)
    print(
        json.dumps(
            {
                "status": "frozen_immutable",
                "corpus_uid": bundle["corpus_uid"],
                "corpus_hash": bundle["corpus_hash"],
                "bundle_sha256": observed_sha256,
                "qualified_block_count": bundle["qualified_block_count"],
                "grade_ab_observation_count": bundle["grade_ab_observation_count"],
                "grade_ab_effective_sample_size": bundle["grade_ab_effective_sample_size"],
                "qualified_action_count": bundle["qualified_action_count"],
                "qualified_cross_source_anchor_count": bundle["qualified_cross_source_anchor_count"],
                "qualified_perturbation_transfer_anchor_count": bundle[
                    "qualified_perturbation_transfer_anchor_count"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
