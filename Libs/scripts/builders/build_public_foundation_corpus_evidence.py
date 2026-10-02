"""Build the identity-bound Phase 1 public-corpus evidence bundle.

This builder materializes only evidence and disposition records.  It never
promotes a legacy matrix, an aggregate atlas prior, or a generated trajectory
to a canonical observation without the V1 qualification contract.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Mapping

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_public_foundation_corpus_legacy_recertification import (
    LEDGER_FIELDS,
    build as build_legacy_recertification,
)
from Libs.utils.data.public_foundation_corpus import CORE_FAMILIES
from Libs.utils.data.public_foundation_evidence import (
    EVIDENCE_PRODUCER,
    EVIDENCE_SCHEMA_VERSION,
    adversarial_leakage_test,
    canonical_hash,
    count_tsv_rows,
    evidence_reference,
    family_assessment_from_rows,
    overlap_counts,
    relative_repo_path,
    safe_repo_path,
    sha256_file,
)
from Libs.utils.data.public_foundation_corpus import hash_path
from Libs.utils.data.public_foundation_storage import (
    pending_storage_ledger,
    storage_gate_status,
)
from Libs.utils.data.raw_data_registry import load_config, read_tsv, write_json, write_tsv


AUDIT_ROOT = REPO_ROOT / "Output/audits"
BUNDLE_PATH = AUDIT_ROOT / "public_foundation_corpus_evidence_bundle.json"
LEGACY_PATH = AUDIT_ROOT / "public_foundation_corpus_legacy_recertification.json"
CROSSWALK_PATH = AUDIT_ROOT / "public_foundation_corpus_entity_unit_crosswalk.json"
CROSSWALK_TABLE_PATH = AUDIT_ROOT / "public_foundation_corpus_entity_unit_crosswalk.tsv"
SOURCE_PATH = AUDIT_ROOT / "public_foundation_corpus_source_qualification.json"
OBSERVATION_PATH = AUDIT_ROOT / "public_foundation_corpus_canonical_observation_blocks.json"
ACTION_PATH = AUDIT_ROOT / "public_foundation_corpus_structured_action_catalog.json"
TOOL_PATH = AUDIT_ROOT / "public_foundation_corpus_tool_model_registry.json"
SPLIT_PATH = AUDIT_ROOT / "public_foundation_corpus_frozen_splits.json"
SPLIT_MANIFEST_PATH = AUDIT_ROOT / "public_foundation_corpus_split_manifest.json"
LEAKAGE_PATH = AUDIT_ROOT / "public_foundation_corpus_leakage_audit.json"
QC_PATH = AUDIT_ROOT / "public_foundation_corpus_qc_ess_statistics.json"
STORAGE_PATH = AUDIT_ROOT / "public_foundation_corpus_storage_ledger.json"
ISOLATION_PATH = AUDIT_ROOT / "public_mainline_isolation_audit.json"

ARTIFACT_VERSION = "neuronsifter-public-foundation-corpus-evidence-artifact-v1"
ACTION_FIELDS = (
    "action_uid",
    "intervention_type",
    "target_set",
    "agent_set",
    "dose_value",
    "dose_unit",
    "schedule",
    "route",
    "assay_context",
    "state_scope",
    "source_publication",
    "source_accession",
    "exposure_program_ref",
    "occupancy_program_ref",
)

SOURCE_SPECS = (
    ("arrayexpress", "Input/raw/arrayexpress/manifests/source_manifest.json", "Input/raw/arrayexpress/terms/biostudies_arrayexpress_snapshot.html"),
    ("geo", "Input/raw/geo/manifests/source_manifest.json", "Input/raw/geo/terms/geo_download_terms.html"),
    ("allen_brain_cell_atlas", "Input/raw/allen_brain_cell_atlas/manifests/source_manifest.json", "Input/raw/allen_brain_cell_atlas/terms/allen_terms_snapshot.html"),
    ("biccn_public", "Input/raw/biccn_public/manifests/source_manifest.json", ""),
    ("clue_lincs", "Input/raw/clue_lincs/manifests/source_manifest.json", "Input/raw/clue_lincs/terms/clue_terms_snapshot.html"),
    ("drugcomb", "Input/raw/drugcomb/manifests/source_manifest.json", "Input/raw/drugcomb/terms/drugcomb_publication_terms_snapshot.html"),
    ("bindingdb", "Input/raw/bindingdb/manifests/source_manifest.json", "Input/raw/bindingdb/terms/bindingdb_home_snapshot.html"),
    ("cellxgene_census", "Input/raw/cellxgene_census/manifests/source_manifest.json", ""),
    ("human_cell_atlas", "Input/raw/human_cell_atlas/manifests/source_manifest.json", ""),
    ("human_protein_atlas", "Input/raw/human_protein_atlas/manifests/source_manifest.json", ""),
    ("jump_cell_painting", "Input/raw/jump_cell_painting/manifests/source_manifest.json", ""),
    ("metabolomics_workbench", "Input/raw/metabolomics_workbench/manifests/source_manifest.json", "Input/raw/metabolomics_workbench/terms/mw_rest_snapshot.html"),
    ("nemo_archive", "Input/raw/nemo_archive/manifests/source_manifest.json", "Input/raw/nemo_archive/terms/nemo_terms_snapshot.html"),
    ("neuromorpho", "Input/raw/neuromorpho/manifests/source_manifest.json", ""),
    ("openneuro", "Input/raw/openneuro/manifests/source_manifest.json", ""),
    ("sra", "Input/raw/sra/manifests/source_manifest.json", ""),
)

# Phase 1A is a source-governance release, not a silent promotion of legacy
# matrices.  Every candidate gets a terminal disposition; only the explicitly
# qualified roles may be consumed by Phase 1A calibration/prior paths.
PHASE1A_DECISIONS: dict[str, tuple[str, str]] = {
    "arrayexpress": ("deferred", "legacy_expression_mirror_requires_semantic_requalification"),
    "geo": ("deferred", "legacy_expression_requires_semantic_requalification"),
    "allen_brain_cell_atlas": ("qualified_as_context_prior", "aggregate_atlas_prior_with_research_only_terms"),
    "biccn_public": ("deferred", "public_routing_metadata_without_native_value_matrix"),
    "clue_lincs": ("qualified_for_weak_calibration", "research_group_restricted_signature_adapter"),
    "drugcomb": ("qualified_for_weak_calibration", "method_pretraining_only_processed_adapter"),
    "bindingdb": ("qualified_for_weak_calibration", "public_affinity_reference_adapter"),
    "cellxgene_census": ("deferred", "candidate_not_materialized_in_v1_release"),
    "human_cell_atlas": ("deferred", "candidate_not_materialized_in_v1_release"),
    "human_protein_atlas": ("deferred", "deferred_modality_not_in_v1_core"),
    "jump_cell_painting": ("deferred", "deferred_modality_not_in_v1_core"),
    "metabolomics_workbench": ("deferred", "deferred_modality_and_legacy_semantic_repair_required"),
    "nemo_archive": ("formally_excluded", "historical_optional_source_outside_v1_release_scope"),
    "neuromorpho": ("deferred", "deferred_modality_not_in_v1_core"),
    "openneuro": ("deferred", "deferred_modality_not_in_v1_core"),
    "sra": ("deferred", "candidate_not_materialized_in_v1_release"),
}
PHASE1A_PROCESSED_BINDINGS = {
    "allen_brain_cell_atlas": "Input/processed/atlas_prior/allen_biccn/program_catalog.tsv",
    "clue_lincs": "Input/processed/intervention_program/intervention_program_catalog.tsv",
    "drugcomb": "Input/processed/external_adapters/drugcomb_combination_prior_catalog.tsv",
    "bindingdb": "Input/processed/drug_target_ready/drug_target_edge_catalog.tsv",
}
PHASE1A_LEDGER_PATH = AUDIT_ROOT / "source_qualification_ledger.tsv"
PHASE1A_INDEPENDENCE_PATH = AUDIT_ROOT / "source_independence_graph.json"
PHASE1A_LICENSE_PATH = AUDIT_ROOT / "license_role_decisions.json"
PHASE1A_RELEASE_PATH = AUDIT_ROOT / "source_release_manifest.tsv"
PHASE1A_MIRROR_PATH = AUDIT_ROOT / "mirror_group_registry.tsv"
PHASE1A_FROZEN_ARTIFACTS = (
    SOURCE_PATH,
    PHASE1A_LEDGER_PATH,
    PHASE1A_INDEPENDENCE_PATH,
    PHASE1A_LICENSE_PATH,
    PHASE1A_RELEASE_PATH,
    PHASE1A_MIRROR_PATH,
)
PHASE1A_LEDGER_FIELDS = (
    "source_id", "source_name", "decision", "status", "decision_reason", "phase1a_consumption",
    "license_status", "role_status", "release_status", "official_url", "release_version",
    "retrieval_date", "source_card_path", "terms_snapshot_path", "terms_snapshot_sha256",
    "license_snapshot_path", "license_snapshot_sha256", "raw_sha256", "processed_sha256",
    "parser_id", "parser_version", "mapping_version", "provenance_record_id",
    "training_evaluation_role", "holdout_group", "publication_accession_group",
    "access_class", "mirror_group", "terminal_status",
)
PHASE1A_RELEASE_FIELDS = (
    "source_id", "release_version", "retrieval_date", "official_url", "source_card_path",
    "terms_snapshot_path", "terms_snapshot_sha256", "raw_sha256", "processed_sha256",
    "parser_id", "parser_version", "mapping_version", "provenance_record_id",
    "release_status", "decision",
)
PHASE1A_MIRROR_FIELDS = (
    "mirror_group", "source_id", "source_accession", "publication_accession_group",
    "relation_type", "independence_status", "provenance_status",
)
P2_T03_OBSERVATION_SOURCE_IDS = frozenset(
    {
        "allen_brain_cell_atlas",
        "arrayexpress",
        "biccn_public",
        "cellxgene_census",
        "geo",
        "human_cell_atlas",
        "human_protein_atlas",
        "jump_cell_painting",
        "metabolomics_workbench",
        "nemo_archive",
        "neuromorpho",
        "openneuro",
        "sra",
    }
)

PHASE1A_ROLE_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "allen_brain_cell_atlas": (
        "gene", "cell_type", "brain_region", "species", "source_release",
    ),
    "clue_lincs": (
        "gene", "compound", "cell_type", "assay", "dose_unit", "time_unit",
        "source_accession",
    ),
    "drugcomb": (
        "compound", "cell_type", "assay", "concentration_unit", "study",
        "source_release",
    ),
    "bindingdb": (
        "compound", "protein", "assay", "affinity_unit", "publication",
        "source_release",
    ),
}

# UCUM spellings observed in the admitted source snapshots.  Unknown values
# are retained as unresolved rows and never silently coerced.
UNIT_MAP: dict[str, tuple[str, str]] = {
    "nM": ("nM", "identity"),
    "uM": ("uM", "identity"),
    "µM": ("uM", "replace_micro_prefix"),
    "ng/mL": ("ng/mL", "identity"),
    "ng/µL": ("ng/uL", "replace_micro_prefix"),
    "ng": ("ng", "identity"),
    "µL": ("uL", "replace_micro_prefix"),
    "h": ("h", "identity"),
    "%": ("%", "identity"),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_artifact(path: Path, payload: dict[str, object]) -> None:
    output = dict(payload)
    output.setdefault("schema_version", ARTIFACT_VERSION)
    output.setdefault("producer", EVIDENCE_PRODUCER)
    output.setdefault("created_at", _now())
    write_json(path, output)


def _read_json(repo_root: Path, relative: str) -> dict[str, object]:
    path = safe_repo_path(repo_root, relative)
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _binding(repo_root: Path, relative: str) -> dict[str, object]:
    path = safe_repo_path(repo_root, relative)
    return {
        "path": relative,
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def _checksum_rows(repo_root: Path, source_id: str) -> tuple[str, int, int]:
    checksum_path = safe_repo_path(repo_root, f"Input/raw/{source_id}/checksums/sha256.tsv")
    if not checksum_path.is_file():
        return "", 0, 0
    rows = read_tsv(checksum_path)
    valid = [row for row in rows if re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256", "")))]
    return sha256_file(checksum_path), len(rows), sum(int(row.get("size_bytes") or 0) for row in valid)


def build_source_qualification(repo_root: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    blockers: set[str] = set()
    for source_id, manifest_rel, terms_rel in SOURCE_SPECS:
        manifest_path = safe_repo_path(repo_root, manifest_rel)
        manifest = _read_json(repo_root, manifest_rel)
        terms_binding = _binding(repo_root, terms_rel) if terms_rel else {
            "path": "",
            "exists": False,
            "size_bytes": 0,
            "sha256": None,
        }
        checksum_sha, checksum_count, declared_bytes = _checksum_rows(repo_root, source_id)
        missing = []
        if not manifest_path.is_file():
            missing.append("source_card")
        if not terms_binding["exists"]:
            missing.append("license_or_terms_snapshot")
        if checksum_count == 0:
            missing.append("checksum_ledger")
        row_blockers = []
        if missing:
            row_blockers.append("required_source_evidence_missing")
        blockers.update(row_blockers)
        rows.append(
            {
                "source_id": source_id,
                "qualification_scope": "p2_t03_observation_candidate"
                if source_id in P2_T03_OBSERVATION_SOURCE_IDS
                else "action_or_auxiliary_candidate",
                "official_url": manifest.get("source_url", ""),
                "release_version": manifest.get("release_label", ""),
                "retrieval_date": manifest.get("download_date", ""),
                "source_card_path": manifest_rel,
                "terms_snapshot_path": terms_rel,
                "terms_snapshot_sha256": terms_binding.get("sha256"),
                "license_snapshot_path": terms_rel,
                "license_snapshot_sha256": terms_binding.get("sha256"),
                "raw_sha256": checksum_sha or None,
                "processed_sha256": None,
                "parser_id": "phase1a_governance_pending",
                "parser_version": "pending",
                "mapping_version": "public_foundation_corpus_v1_pending",
                "provenance_record_id": "",
                "training_evaluation_role": "candidate_quarantine",
                "holdout_group": "",
                "publication_accession_group": "",
                "access_class": "public_with_attribution"
                if manifest.get("access_mode") == "public"
                else "unqualified_pending",
                "checksum_ledger_path": f"Input/raw/{source_id}/checksums/sha256.tsv",
                "checksum_ledger_sha256": checksum_sha or None,
                "checksum_row_count": checksum_count,
                "declared_asset_bytes": declared_bytes,
                "status": "pending",
                "blocker_codes": sorted(set(row_blockers)),
            }
        )
    return {
        "status": "pending",
        "record_count": len(rows),
        "source_count": len(rows),
        "p2_t03_observation_candidate_count": sum(
            row["qualification_scope"] == "p2_t03_observation_candidate" for row in rows
        ),
        "action_or_auxiliary_candidate_count": sum(
            row["qualification_scope"] == "action_or_auxiliary_candidate" for row in rows
        ),
        "qualified_source_count": 0,
        "critical_mapping_conflict_count": 0,
        "sources": rows,
        "blocker_codes": sorted(blockers),
    }


def _phase1a_processed_binding(repo_root: Path, source_id: str) -> tuple[str, str]:
    relative = PHASE1A_PROCESSED_BINDINGS.get(source_id)
    if not relative:
        return "not_applicable_phase1a", "not_applicable"
    path = safe_repo_path(repo_root, relative)
    if not path.is_file():
        return "not_available_phase1a", "not_available"
    return sha256_file(path), relative


def build_phase1a_governance(
    repo_root: Path,
    source: Mapping[str, object],
    blocks: list[Mapping[str, object]],
    artifact_root: Path | None = None,
) -> dict[str, object]:
    """Close source decisions without promoting observation candidates.

    The source qualification status is intentionally independent from G2: a
    source can be terminally deferred or excluded while the observation family
    remains unqualified.
    """
    artifact_root = (artifact_root or AUDIT_ROOT).resolve()
    source_rows = list(source.get("sources", []))
    block_groups: dict[str, list[Mapping[str, object]]] = {}
    for block in blocks:
        block_groups.setdefault(str(block.get("source_id", "")), []).append(block)
    ledger_rows: list[dict[str, object]] = []
    release_rows: list[dict[str, object]] = []
    mirror_rows: list[dict[str, object]] = []
    license_decisions: list[dict[str, object]] = []

    for row in source_rows:
        source_id = str(row["source_id"])
        decision, reason = PHASE1A_DECISIONS[source_id]
        actual_role = decision in {"qualified_for_mainline", "qualified_for_weak_calibration", "qualified_as_context_prior"}
        terms_exists = bool(row.get("terms_snapshot_sha256"))
        processed_sha, processed_ref = _phase1a_processed_binding(repo_root, source_id)
        if actual_role:
            license_status = "qualified_terms_snapshot_recorded" if terms_exists else "license_unresolved"
            role_status = "qualified_role_declared"
            release_status = "qualified_release_recorded" if row.get("release_version") and row.get("retrieval_date") else "release_unresolved"
            parser_id = {
                "allen_brain_cell_atlas": "build_atlas_prior_registry",
                "clue_lincs": "build_intervention_program_registry",
                "drugcomb": "build_external_adapter_processed_catalogs",
                "bindingdb": "build_drug_target_ready_registry",
            }.get(source_id, "phase1a_source_parser")
            parser_version = "phase1a_parser_contract_v1"
            mapping_version = {
                "allen_brain_cell_atlas": "allen_seaad_program_mapping_v1",
                "clue_lincs": "clue_lincs_signature_mapping_v1",
                "drugcomb": "drugcomb_response_mapping_v1",
                "bindingdb": "bindingdb_affinity_mapping_v1",
            }.get(source_id, "public_foundation_corpus_v1_mapping_v1")
            provenance_id = f"phase1a:{source_id}:public_snapshot"
            phase1a_consumption = "weak_calibration" if decision == "qualified_for_weak_calibration" else "context_prior" if decision == "qualified_as_context_prior" else "mainline"
        else:
            license_status = "not_applicable_not_consumed"
            role_status = "deferred_not_consumed" if decision == "deferred" else "formally_excluded_audit_only"
            release_status = "deferred_release_recorded" if row.get("release_version") else "not_applicable_not_consumed"
            parser_id = "not_applicable_phase1a_deferred"
            parser_version = "phase1a_governance_v1"
            mapping_version = "not_applicable_phase1a_deferred"
            provenance_id = f"phase1a:decision:{source_id}"
            phase1a_consumption = "not_used_phase1a" if decision == "deferred" else "excluded_audit_only"
            processed_sha = "not_applicable_phase1a" if processed_sha == "not_available_phase1a" else processed_sha
        unresolved = [code for code, value in (("license_unresolved", license_status), ("role_unresolved", role_status), ("release_unresolved", release_status)) if value in {"license_unresolved", "role_unresolved", "release_unresolved"}]
        if not actual_role:
            unresolved = []
        group_rows = block_groups.get(source_id, [])
        source_groups = sorted({str(b.get("publication_accession_group")) for b in group_rows if b.get("publication_accession_group")})
        mirror_groups = sorted({str(b.get("mirror_group")) for b in group_rows if b.get("mirror_group")})
        publication_group = "|".join(source_groups) if source_groups else f"source_release::{source_id}::{row.get('release_version') or 'not_materialized'}"
        mirror_group = "|".join(mirror_groups) if mirror_groups else f"source::{source_id}"
        terminal = decision in {"qualified_for_mainline", "qualified_for_weak_calibration", "qualified_as_context_prior", "deferred", "formally_excluded"}
        ledger = {
            **row,
            "source_name": _read_json(repo_root, str(row["source_card_path"])).get("source_name", source_id),
            "decision": decision,
            "status": decision,
            "decision_reason": reason,
            "phase1a_consumption": phase1a_consumption,
            "license_status": license_status,
            "role_status": role_status,
            "release_status": release_status,
            "processed_sha256": processed_sha,
            "processed_reference": processed_ref,
            "parser_id": parser_id,
            "parser_version": parser_version,
            "mapping_version": mapping_version,
            "provenance_record_id": provenance_id,
            "training_evaluation_role": phase1a_consumption,
            "holdout_group": f"phase1a_source_holdout::{source_id}",
            "publication_accession_group": publication_group,
            "mirror_group": mirror_group,
            "terminal_status": "terminal" if terminal else "unresolved",
            "blocker_codes": (["license_unresolved"] if "license_unresolved" in unresolved else []) + (["release_unresolved"] if "release_unresolved" in unresolved else []),
        }
        ledger_rows.append(ledger)
        release_rows.append({key: ledger.get(key, "") for key in PHASE1A_RELEASE_FIELDS})
        license_decisions.append({
            "source_id": source_id,
            "decision": decision,
            "license_status": license_status,
            "role_status": role_status,
            "release_status": release_status,
            "terms_snapshot_sha256": ledger.get("terms_snapshot_sha256"),
            "use_scope": "internal_research_only" if source_id in {"allen_brain_cell_atlas", "clue_lincs"} else "public_research_with_attribution" if actual_role else "not_consumed",
            "redistribution_allowed": False if source_id in {"allen_brain_cell_atlas", "clue_lincs"} else None if not actual_role else True,
            "raw_asset_release_allowed": False,
            "derived_artifact_release_allowed": False if source_id in {"allen_brain_cell_atlas", "clue_lincs"} else True if actual_role else False,
            "role_boundary": "context_prior_only" if decision == "qualified_as_context_prior" else "weak_calibration_only" if decision == "qualified_for_weak_calibration" else "not_consumed",
            "decision_reason": reason,
            "consumed_by_phase1a": actual_role,
        })
        for block in group_rows:
            mirror_rows.append({
                "mirror_group": block.get("mirror_group") or mirror_group,
                "source_id": source_id,
                "source_accession": block.get("source_accession", ""),
                "publication_accession_group": block.get("publication_accession_group", publication_group),
                "relation_type": "mirror_candidate" if str(block.get("mirror_group", "")).startswith("geo::") and source_id == "arrayexpress" else "source_membership",
                "independence_status": "not_independent_mirror" if str(block.get("mirror_group", "")).startswith("geo::") and source_id == "arrayexpress" else "independent_candidate_group",
                "provenance_status": "frozen_from_canonical_observation_block",
            })
        if not group_rows:
            mirror_rows.append({
                "mirror_group": mirror_group,
                "source_id": source_id,
                "source_accession": "",
                "publication_accession_group": publication_group,
                "relation_type": "source_release_membership",
                "independence_status": "not_applicable_no_value_block" if decision in {"deferred", "formally_excluded"} else "independent_candidate_group",
                "provenance_status": "frozen_from_source_release_manifest",
            })

    unresolved_count = sum(1 for row in ledger_rows for code in row.get("blocker_codes", []) if code in {"license_unresolved", "role_unresolved", "release_unresolved"})
    decisions = {key: sum(row["decision"] == key for row in ledger_rows) for key in ("qualified_for_mainline", "qualified_for_weak_calibration", "qualified_as_context_prior", "deferred", "formally_excluded")}
    source_out = {
        **source,
        "status": "passed" if unresolved_count == 0 and all(row["terminal_status"] == "terminal" for row in ledger_rows) else "not_passed",
        "phase1a_status": "passed" if unresolved_count == 0 and all(row["terminal_status"] == "terminal" for row in ledger_rows) else "not_passed",
        "qualified_source_count": sum(row["decision"].startswith("qualified_") for row in ledger_rows),
        "terminal_source_count": sum(row["terminal_status"] == "terminal" for row in ledger_rows),
        "decision_counts": decisions,
        "unresolved_counts": {"license_unresolved": sum(row["license_status"] == "license_unresolved" for row in ledger_rows), "role_unresolved": sum(row["role_status"] == "role_unresolved" for row in ledger_rows), "release_unresolved": sum(row["release_status"] == "release_unresolved" for row in ledger_rows)},
        "blocker_codes": [] if unresolved_count == 0 else ["source_governance_unresolved"],
        "sources": ledger_rows,
        "phase1a_artifacts": {
            "source_qualification_ledger": str((artifact_root / PHASE1A_LEDGER_PATH.name).relative_to(repo_root)),
            "source_independence_graph": str((artifact_root / PHASE1A_INDEPENDENCE_PATH.name).relative_to(repo_root)),
            "license_role_decisions": str((artifact_root / PHASE1A_LICENSE_PATH.name).relative_to(repo_root)),
            "source_release_manifest": str((artifact_root / PHASE1A_RELEASE_PATH.name).relative_to(repo_root)),
            "mirror_group_registry": str((artifact_root / PHASE1A_MIRROR_PATH.name).relative_to(repo_root)),
        },
    }
    graph_nodes = [{"source_id": row["source_id"], "decision": row["decision"], "publication_accession_group": row["publication_accession_group"], "mirror_group": row["mirror_group"]} for row in ledger_rows]
    graph_edges = [{"source_id": row["source_id"], "mirror_group": row["mirror_group"], "relation": "source_membership"} for row in ledger_rows]
    graph = {"schema_version": "neuronsifter-phase1a-source-independence-graph-v1", "status": "passed", "nodes": graph_nodes, "edges": graph_edges, "mirror_groups_frozen": True, "publication_accession_groups_frozen": True}
    license_artifact = {"schema_version": "neuronsifter-phase1a-license-role-decisions-v1", "status": "passed" if unresolved_count == 0 else "not_passed", "decision_count": len(license_decisions), "unresolved_counts": source_out["unresolved_counts"], "decisions": license_decisions}
    return {"source": source_out, "ledger_rows": ledger_rows, "release_rows": release_rows, "mirror_rows": mirror_rows, "graph": graph, "license": license_artifact}


def _study_rows(repo_root: Path) -> dict[str, dict[str, str]]:
    path = safe_repo_path(repo_root, "Input/processed/model_input/study_bundle_catalog.tsv")
    return {row["study_uid"]: row for row in read_tsv(path)} if path.is_file() else {}


def _declared_anchor(repo_root: Path, relative: str, source_id: str) -> tuple[str | None, int]:
    path = safe_repo_path(repo_root, f"Input/raw/{source_id}/checksums/sha256.tsv")
    if not path.is_file():
        return None, 0
    rows = read_tsv(path)
    match = next((row for row in rows if row.get("relative_path") == relative), None)
    if match is None:
        return None, 0
    return str(match.get("sha256") or "") or None, int(match.get("size_bytes") or 0)


def build_observation_blocks(
    repo_root: Path,
    *,
    include_phase1e_re_admission: bool = True,
) -> dict[str, object]:
    blocks: list[dict[str, object]] = []
    studies = _study_rows(repo_root)
    aligned_path = safe_repo_path(repo_root, "Input/processed/alignment_ready/aligned_matrix_catalog.tsv")
    if aligned_path.is_file():
        for row in read_tsv(aligned_path):
            accession = row["source_accession"]
            matrix_rel = row["matrix_raw_path"]
            matrix_path = safe_repo_path(repo_root, matrix_rel)
            semantic_conflict = accession == "E-GEOD-6613"
            blocks.append(
                {
                    "canonical_uid": f"migration_{row['study_uid']}",
                    "family_id": "cns_disease_aging_state_transcriptomics",
                    "source_id": row["source_key"],
                    "source_accession": accession,
                    "publication_accession_group": f"{row['source_key']}::{accession}",
                    "mirror_group": f"geo::{accession}",
                    "cns_source": row.get("tissue_or_biofluid", "").lower() not in {"whole blood", "blood"},
                    "biological_unit_count": int(row.get("sample_count") or 0),
                    "observation_count": int(row.get("sample_count") or 0),
                    "effective_sample_size": 0.0,
                    "value_block_format": "npy_migration_input",
                    "accepted_value_block_format": False,
                    "value_block_path": matrix_rel,
                    "value_bytes": matrix_path.stat().st_size if matrix_path.is_file() else 0,
                    "block_sha256": sha256_file(matrix_path) if matrix_path.is_file() else None,
                    "qc_grade": "D" if semantic_conflict else "C",
                    "mapping_confidence": "partial_gene_only",
                    "ontology_versions": {},
                    "unit_mapping_version": "unresolved",
                    "split_role": "quarantine",
                    "observation_eligible": False,
                    "semantic_axes_complete": False,
                    "blocker_codes": [
                        "raw_matrix_migration_input_only",
                        "source_license_pending",
                        "publication_accession_provenance_pending",
                    ] + (["pd_study_mislabeled_as_ad", "whole_blood_region_conflict"] if semantic_conflict else []),
                }
            )
    for study_uid, row in sorted(studies.items()):
        if row.get("source_accession") == "GSE13214":
            matrix_rel = row["matrix_raw_path"]
            matrix_path = safe_repo_path(repo_root, matrix_rel)
            blocks.append(
                {
                    "canonical_uid": f"migration_{study_uid}",
                    "family_id": "cns_disease_aging_state_transcriptomics",
                    "source_id": row["source_key"],
                    "source_accession": row["source_accession"],
                    "publication_accession_group": f"{row['source_key']}::{row['source_accession']}",
                    "mirror_group": f"geo::{row['source_accession']}",
                    "cns_source": True,
                    "biological_unit_count": int(row.get("sample_count") or 0),
                    "observation_count": int(row.get("sample_count") or 0),
                    "effective_sample_size": 0.0,
                    "value_block_format": "npy_migration_input",
                    "accepted_value_block_format": False,
                    "value_block_path": matrix_rel,
                    "value_bytes": matrix_path.stat().st_size if matrix_path.is_file() else 0,
                    "block_sha256": sha256_file(matrix_path) if matrix_path.is_file() else None,
                    "qc_grade": "C",
                    "mapping_confidence": "unresolved_gene_alignment",
                    "ontology_versions": {},
                    "unit_mapping_version": "unresolved",
                    "split_role": "quarantine",
                    "observation_eligible": False,
                    "semantic_axes_complete": False,
                    "blocker_codes": ["gene_alignment_missing", "source_license_pending"],
                }
            )
    allen_hash, allen_bytes = _declared_anchor(
        repo_root,
        "downloads/releases/20250531/SEAAD/precomputed_stats.20231120.sea_ad.MTG.h5",
        "allen_brain_cell_atlas",
    )
    blocks.append(
        {
            "canonical_uid": "candidate_allen_seaad_mtg_20250531",
            "family_id": "cns_cell_type_region_atlas_transcriptomics",
            "source_id": "allen_brain_cell_atlas",
            "source_accession": "SEAAD-20250531",
            "publication_accession_group": "allen::SEAAD-20250531",
            "mirror_group": "allen::SEAAD-20250531",
            "cns_source": True,
            "biological_unit_count": 0,
            "observation_count": 0,
            "effective_sample_size": 0.0,
            "value_block_format": "hdf5_aggregate_prior",
            "accepted_value_block_format": False,
            "value_block_path": "Input/raw/allen_brain_cell_atlas/downloads/releases/20250531/SEAAD/precomputed_stats.20231120.sea_ad.MTG.h5",
            "value_bytes": allen_bytes,
            "block_sha256": allen_hash,
            "qc_grade": "C",
            "mapping_confidence": "aggregate_prior_only",
            "ontology_versions": {},
            "unit_mapping_version": "not_applicable",
            "split_role": "quarantine",
            "observation_eligible": False,
            "semantic_axes_complete": False,
            "blocker_codes": ["aggregate_programs_not_native_observations", "second_independent_source_missing"],
        }
    )
    clue_hash, clue_bytes = _declared_anchor(
        repo_root,
        "downloads/builds/TouchstoneV1.1/files/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx",
        "clue_lincs",
    )
    blocks.append(
        {
            "canonical_uid": "candidate_clue_touchstone_v1_1",
            "family_id": "compound_dose_time_perturbation_transcriptomics",
            "source_id": "clue_lincs",
            "source_accession": "GSE92742-TouchstoneV1.1",
            "publication_accession_group": "clue::GSE92742-TouchstoneV1.1",
            "mirror_group": "clue::GSE92742-TouchstoneV1.1",
            "cns_source": False,
            "biological_unit_count": 0,
            "observation_count": 473647,
            "effective_sample_size": 0.0,
            "value_block_format": "gctx_signature_matrix",
            "accepted_value_block_format": False,
            "value_block_path": "Input/raw/clue_lincs/downloads/builds/TouchstoneV1.1/files/GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx",
            "value_bytes": clue_bytes,
            "block_sha256": clue_hash,
            "qc_grade": "C",
            "mapping_confidence": "non_cns_signature_subset_unqualified",
            "ontology_versions": {},
            "unit_mapping_version": "unresolved",
            "split_role": "quarantine",
            "observation_eligible": False,
            "semantic_axes_complete": False,
            "blocker_codes": ["qualified_cns_subset_missing", "dose_time_canonicalization_pending", "second_independent_source_missing"],
        }
    )

    # Phase 1A decisions remain source-wide and frozen.  Each Phase 1E block
    # is accession-scoped, validator-gated, and must bind an admission audit;
    # merely finding a matrix on disk never promotes it.
    if include_phase1e_re_admission:
        config = load_config(safe_repo_path(repo_root, "Libs/configs/config_stack.yaml"))
        phase1e_specs = (
            (
                "phase1e_materialization",
                "Libs.scripts.validators.validate_phase1e_gse28146_mainline_block",
                "validate",
            ),
            (
                "phase1f_disease_pair_materialization",
                "Libs.scripts.validators.validate_phase1f_gse37263_mainline_block",
                "validate",
            ),
            (
                "phase1e_atlas_materialization",
                "Libs.scripts.validators.validate_phase1e_allen_whb_nonneurons_mainline_block",
                "validate",
            ),
            (
                "phase1e_cellxgene_atlas_materialization",
                "Libs.scripts.validators.validate_phase1e_cellxgene_amygdala_mainline_block",
                "validate",
            ),
            (
                "phase1e_perturbation_materialization",
                "Libs.scripts.validators.validate_phase1e_clue_npc_neu_mainline_block",
                "validate",
            ),
            (
                "phase1f_perturbation_pair_materialization",
                "Libs.scripts.validators.validate_phase1f_perturbation_pair_block",
                "validate",
            ),
        )
        for policy_key, module_name, validator_name in phase1e_specs:
            phase1e_policy = config["public_foundation_corpus"][policy_key]
            admission_path = safe_repo_path(repo_root, str(phase1e_policy["admission_audit_path"]))
            if not admission_path.is_file():
                # The candidate has not completed a materialization attempt;
                # it remains a separately recorded non-mainline candidate.
                continue
            module = __import__(module_name, fromlist=[validator_name])
            phase1e_report = getattr(module, validator_name)(repo_root)
            if phase1e_report.get("status") != "passed":
                raise ValueError(
                    f"Phase 1E {policy_key} mainline block is not validated: "
                    + json.dumps(phase1e_report.get("blockers", []), sort_keys=True)
                )
            admission = _read_json(repo_root, str(phase1e_policy["admission_audit_path"]))
            manifest_binding = admission.get("canonical_block_manifest")
            if not isinstance(manifest_binding, Mapping):
                raise ValueError("Phase 1E admission audit lacks a canonical block manifest binding")
            manifest_path = safe_repo_path(repo_root, str(manifest_binding.get("path", "")))
            if not manifest_path.is_file() or sha256_file(manifest_path) != manifest_binding.get("sha256"):
                raise ValueError("Phase 1E admission audit does not bind the current canonical manifest")
            manifest = _read_json(repo_root, relative_repo_path(repo_root, manifest_path))
            block_root = manifest_path.parent
            values_path = block_root / str(manifest.get("value_block_path", ""))
            if not values_path.exists():
                raise ValueError("Phase 1E canonical value block is missing")
            _, value_bytes, value_hash = hash_path(values_path)
            if not value_bytes or not value_hash:
                raise ValueError("Phase 1E canonical value block has no content-addressed bytes")
            blocks.append(
                {
                    "canonical_uid": manifest["canonical_uid"],
                    "family_id": manifest["family_id"],
                    "source_id": manifest["source_id"],
                    "source_accession": manifest["source_accession"],
                    "publication_accession_group": manifest["publication_accession_group"],
                    "mirror_group": manifest["mirror_group"],
                    "cns_source": manifest["cns_source"],
                    "biological_unit_count": manifest["biological_unit_count"],
                    "observation_count": manifest["observation_count"],
                    "effective_sample_size": manifest["effective_sample_size"],
                    "value_block_format": manifest["value_block_format"],
                    "accepted_value_block_format": manifest["value_block_format"] in {"zarr", "parquet", "h5ad"},
                    "value_block_path": relative_repo_path(repo_root, values_path),
                    "value_bytes": value_bytes,
                    "block_sha256": manifest["block_sha256"],
                    "value_block_sha256": value_hash,
                    "qc_grade": manifest["qc_grade"],
                    "mapping_confidence": manifest["mapping_confidence"],
                    "ontology_versions": manifest["ontology_versions"],
                    "unit_mapping_version": manifest["unit_mapping_version"],
                    "split_role": manifest["split_role"],
                    "observation_eligible": True,
                    "semantic_axes_complete": manifest["semantic_axes_complete"],
                    "role": manifest["role"],
                    "admission_allowed": manifest["admission_allowed"],
                    "manifest_path": relative_repo_path(repo_root, manifest_path),
                    "manifest_sha256": sha256_file(manifest_path),
                    "validation_path": str(phase1e_policy["validation_path"]),
                    "blocker_codes": list(manifest.get("blocker_codes", [])),
                }
            )
    qualified_blocks = [block for block in blocks if block["observation_eligible"] is True]
    # Block manifests retain the blockers that were true when each accession
    # was admitted. Corpus evidence is a current-state surface, so source-pair
    # blockers must be derived from the complete validator-gated block set.
    family_pair_closed = {
        family_id: family_assessment_from_rows(
            family_id,
            qualified_blocks,
            query_domain_status="frozen",
            nested_split_status="frozen",
            leakage_status="passed",
        )["qualified_source_count"] >= 2
        and family_assessment_from_rows(
            family_id,
            qualified_blocks,
            query_domain_status="frozen",
            nested_split_status="frozen",
            leakage_status="passed",
        )["independent_publication_accession_group_count"] >= 2
        for family_id in CORE_FAMILIES
    }
    historical_pair_blockers = {
        "second_independent_disease_source_missing": "cns_disease_aging_state_transcriptomics",
        "disease_second_independent_source_missing": "cns_disease_aging_state_transcriptomics",
        "atlas_second_independent_source_missing": "cns_cell_type_region_atlas_transcriptomics",
        "second_independent_atlas_source_missing": "cns_cell_type_region_atlas_transcriptomics",
        "perturbation_second_independent_source_missing": "compound_dose_time_perturbation_transcriptomics",
        "second_independent_perturbation_source_missing": "compound_dose_time_perturbation_transcriptomics",
    }
    quarantine_blockers = sorted({
        code
        for block in blocks
        for code in block["blocker_codes"]
        if not (
            code in historical_pair_blockers
            and family_pair_closed[historical_pair_blockers[code]]
        )
    })
    gate_blockers = sorted(
        f"{family_id}:independent_pair_not_closed"
        for family_id, closed in family_pair_closed.items()
        if not closed
    )
    return {
        "status": "passed" if not gate_blockers else "not_passed",
        "record_count": len(blocks),
        "accepted_value_block_formats": ["zarr", "parquet", "h5ad"],
        "candidate_real_value_block_count": len(blocks),
        "qualified_real_value_block_count": len(qualified_blocks),
        "synthetic_value_block_count": 0,
        "blocks": blocks,
        "qualified_blocks": qualified_blocks,
        "family_independent_pair_closed": family_pair_closed,
        "quarantine_blocker_codes": quarantine_blockers,
        "blocker_codes": gate_blockers,
    }


def _normalise_token(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _canonical_text(value: object) -> str:
    return _normalise_token(value).casefold()


def _read_gzip_tsv(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="")


def build_crosswalk(repo_root: Path, *, table_path: Path = CROSSWALK_TABLE_PATH) -> dict[str, object]:
    """Materialize role-scoped entity/unit mappings from the four admitted sources.

    This is intentionally an evidence crosswalk, not an observation promotion:
    rows with missing source values are retained as quarantined mappings and
    never coerced to a canonical value.
    """
    required = ("gene", "protein", "compound", "cell_type", "brain_region", "assay", "disease", "publication", "species", "unit")
    table_rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, str, str]] = set()
    quarantined = 0
    ambiguities = 0

    def add(source_id: str, entity_type: str, raw: object, canonical: object, namespace: str,
            *, raw_unit: object = "", canonical_unit: object = "", formula: str = "identity",
            version: str = "public_foundation_corpus_v1_crosswalk_2026-07-17",
            confidence: str = "exact", status: str = "resolved", provenance: str = "") -> None:
        nonlocal quarantined, ambiguities
        raw_text = _normalise_token(raw)
        canonical_text = _normalise_token(canonical)
        if not raw_text or not canonical_text or canonical_text in {"-666", "NA", "N/A", "NULL"}:
            status = "quarantined_unresolved"
            canonical_text = ""
            quarantined += 1
        if status == "ambiguous":
            ambiguities += 1
        key = (source_id, entity_type, raw_text, canonical_text)
        if key in seen:
            return
        seen.add(key)
        table_rows.append({
            "source_id": source_id,
            "entity_type": entity_type,
            "raw_value": raw_text,
            "canonical_value": canonical_text,
            "canonical_namespace": namespace,
            "mapping_status": status,
            "mapping_confidence": confidence,
            "raw_unit": _normalise_token(raw_unit),
            "canonical_unit": _normalise_token(canonical_unit),
            "conversion_formula": formula,
            "ontology_or_unit_version": version,
            "provenance_ref": provenance,
        })

    gene_catalog = safe_repo_path(repo_root, "Input/processed/alignment_ready/entity_catalog.tsv")
    gene_by_label: dict[str, str] = {}
    if gene_catalog.is_file():
        with gene_catalog.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row.get("entity_type") == "gene":
                    gene_by_label[_canonical_text(row.get("canonical_entity_label"))] = str(row.get("canonical_entity_id", ""))
                    add("alignment_backbone", "gene", row.get("canonical_entity_label"), row.get("canonical_entity_id"), "entrez_gene_id", confidence="exact", provenance="Input/processed/alignment_ready/entity_catalog.tsv")

    # Allen aggregate prior: valid for ontology/context only.
    allen = safe_repo_path(repo_root, PHASE1A_PROCESSED_BINDINGS["allen_brain_cell_atlas"])
    add("allen_brain_cell_atlas", "source_release", "SEAAD-20250531", "allen::SEAAD-20250531", "source_release", confidence="provenance_only", provenance="Output/audits/source_release_manifest.tsv")
    if allen.is_file():
        with allen.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                provenance = "allen_brain_cell_atlas::SEAAD-20250531"
                add("allen_brain_cell_atlas", "species", row.get("species"), _canonical_text(row.get("species")), "NCBITaxon", provenance=provenance)
                add("allen_brain_cell_atlas", "brain_region", row.get("brain_region_id"), row.get("brain_region_id"), "UBERON", provenance=provenance)
                add("allen_brain_cell_atlas", "cell_type", row.get("cell_type_id"), row.get("cell_type_id"), "CL", provenance=provenance)
                for feature in str(row.get("feature_ids", "")).split("|"):
                    if feature:
                        add("allen_brain_cell_atlas", "gene", feature, gene_by_label.get(_canonical_text(feature), feature), "entrez_gene_id", confidence="exact" if _canonical_text(feature) in gene_by_label else "source_native", provenance=provenance)

    # WHB-native metadata is retained separately from the SEAAD aggregate
    # prior.  These are exact source identifiers used by the Phase 1E raw-count
    # materializer; they do not turn the aggregate prior into an observation.
    allen_whb_root = safe_repo_path(repo_root, "Input/raw/allen_brain_cell_atlas/downloads/releases/20250531/WHB-10Xv3")
    whb_taxonomy_root = safe_repo_path(repo_root, "Input/raw/allen_brain_cell_atlas/downloads/releases/20250531/WHB-taxonomy")
    whb_provenance = "allen_brain_cell_atlas::WHB-10Xv3-20240330/20241115"
    add("allen_brain_cell_atlas", "source_release", "WHB-10Xv3-20240330", "allen::WHB-10Xv3-20240330", "source_release", confidence="provenance_only", provenance=whb_provenance)
    donor_path = allen_whb_root / "donor.csv"
    if donor_path.is_file():
        with donor_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                add("allen_brain_cell_atlas", "species", row.get("donor_species"), row.get("donor_species"), "NCBITaxon", provenance=whb_provenance)
    gene_path = allen_whb_root / "gene.csv"
    if gene_path.is_file():
        with gene_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                add("allen_brain_cell_atlas", "gene", row.get("gene_identifier"), row.get("gene_identifier"), "Ensembl_Gene", confidence="exact", provenance=whb_provenance)
                add("allen_brain_cell_atlas", "gene", row.get("gene_symbol"), row.get("gene_identifier"), "Ensembl_Gene", confidence="exact", provenance=whb_provenance)
    region_path = allen_whb_root / "region_of_interest_structure_map.csv"
    if region_path.is_file():
        with region_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                add("allen_brain_cell_atlas", "brain_region", row.get("region_of_interest_label"), row.get("structure_identifier"), "DHBA", confidence="exact", provenance=whb_provenance)
    cluster_path = whb_taxonomy_root / "cluster.csv"
    if cluster_path.is_file():
        with cluster_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                add("allen_brain_cell_atlas", "cell_type", row.get("cluster_alias"), row.get("label"), "Allen_WHB_CCN202210140", confidence="exact", provenance=whb_provenance)

    # CLUE/LINCS metadata supplies real perturbagen, cell, assay, dose and time axes.
    clue_root = safe_repo_path(repo_root, "Input/raw/clue_lincs/downloads/builds/TouchstoneV1.1/files")
    add("clue_lincs", "source_accession", "GSE92742-TouchstoneV1.1", "clue::GSE92742-TouchstoneV1.1", "publication_accession_group", confidence="provenance_only", provenance="Output/audits/source_release_manifest.tsv")
    pert_path = clue_root / "GSE92742_Broad_LINCS_pert_info.txt.gz"
    if pert_path.is_file():
        with _read_gzip_tsv(pert_path) as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                add("clue_lincs", "compound", row.get("pert_id"), f"clue:{row.get('pert_id')}", "CLUE_pert_id", confidence="source_native", provenance="clue_lincs::GSE92742-TouchstoneV1.1")
                add("clue_lincs", "assay", row.get("pert_type"), f"clue_assay:{row.get('pert_type')}", "CLUE_pert_type", confidence="source_native", provenance="clue_lincs::GSE92742-TouchstoneV1.1")
    gene_path = clue_root / "GSE92742_Broad_LINCS_gene_info.txt.gz"
    if gene_path.is_file():
        with _read_gzip_tsv(gene_path) as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                symbol = row.get("pr_gene_symbol")
                add("clue_lincs", "gene", symbol, gene_by_label.get(_canonical_text(symbol), f"clue_gene:{row.get('pr_gene_id')}"), "entrez_gene_id", confidence="exact" if _canonical_text(symbol) in gene_by_label else "source_native", provenance="clue_lincs::GSE92742-TouchstoneV1.1")
    sig_path = clue_root / "GSE92742_Broad_LINCS_sig_info.txt.gz"
    if sig_path.is_file():
        with _read_gzip_tsv(sig_path) as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                provenance = "clue_lincs::GSE92742-TouchstoneV1.1"
                add("clue_lincs", "cell_type", row.get("cell_id"), f"clue_cell:{row.get('cell_id')}", "CLUE_cell_id", confidence="source_native", provenance=provenance)
                dose_unit, dose_formula = UNIT_MAP.get(str(row.get("pert_dose_unit")), ("", "unresolved"))
                time_unit, time_formula = UNIT_MAP.get(str(row.get("pert_time_unit")), ("", "unresolved"))
                add("clue_lincs", "unit", row.get("pert_dose_unit"), dose_unit, "UCUM", raw_unit=row.get("pert_dose_unit"), canonical_unit=dose_unit, formula=dose_formula, confidence="exact" if dose_unit else "unresolved", provenance=provenance)
                add("clue_lincs", "unit", row.get("pert_time_unit"), time_unit, "UCUM", raw_unit=row.get("pert_time_unit"), canonical_unit=time_unit, formula=time_formula, confidence="exact" if time_unit else "unresolved", provenance=provenance)
                add("clue_lincs", "assay", row.get("pert_type"), f"clue_assay:{row.get('pert_type')}", "CLUE_pert_type", confidence="source_native", provenance=provenance)

    # DrugComb response adapter: aggregate unique IDs/axes, preserving units.
    drugcomb = safe_repo_path(repo_root, "Input/processed/external_adapters/drugcomb_combination_prior_catalog.tsv")
    add("drugcomb", "source_release", "drugcomb_zenodo_v1_4_v1_5", "source_release::drugcomb::drugcomb_zenodo_v1_4_v1_5", "source_release", confidence="provenance_only", provenance="Output/audits/source_release_manifest.tsv")
    if drugcomb.is_file():
        with drugcomb.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                provenance = f"drugcomb::{row.get('source_release') or 'public_snapshot'}::{row.get('block_id') or 'unknown'}"
                for value, cid in ((row.get("drug_row"), row.get("drug_row_cid")), (row.get("drug_col"), row.get("drug_col_cid"))):
                    if _normalise_token(value) and _normalise_token(value).upper() not in {"NULL", "\\N"}:
                        native = cid if _normalise_token(cid) not in {"", "0", "\\N"} else _canonical_text(value)
                        add("drugcomb", "compound", value, f"drugcomb:{native}", "DrugComb_compound", confidence="source_native", provenance=provenance)
                add("drugcomb", "cell_type", row.get("cell_line_name"), f"drugcomb_cell:{row.get('cell_line_name')}", "DrugComb_cell_line", confidence="source_native", provenance=provenance)
                add("drugcomb", "brain_region", row.get("tissue_name"), f"drugcomb_tissue:{_canonical_text(row.get('tissue_name'))}", "UBERON_or_source_tissue", confidence="source_native", provenance=provenance)
                add("drugcomb", "study", row.get("study_name"), f"drugcomb_study:{_canonical_text(row.get('study_name'))}", "DrugComb_study", confidence="source_native", provenance=provenance)
                add("drugcomb", "publication", row.get("study_name"), f"drugcomb_publication:{_canonical_text(row.get('study_name'))}", "DrugComb_study", confidence="provenance_only", provenance=provenance)
                add("drugcomb", "assay", row.get("primary_synergy_metric"), f"drugcomb_assay:{row.get('primary_synergy_metric')}", "DrugComb_metric", confidence="source_native", provenance=provenance)
                for unit in (row.get("conc_row_unit"), row.get("conc_col_unit")):
                    if _normalise_token(unit).upper() not in {"", "NA", "N/A"}:
                        canonical, formula = UNIT_MAP.get(str(unit), ("", "unresolved"))
                        add("drugcomb", "unit", unit, canonical, "UCUM", raw_unit=unit, canonical_unit=canonical, formula=formula, confidence="exact" if canonical else "unresolved", provenance=provenance)

    # BindingDB affinity edges bind compounds, targets/proteins, assay, units,
    # disease context and publication provenance.
    binding = safe_repo_path(repo_root, "Input/processed/drug_target_ready/drug_target_edge_catalog.tsv")
    add("bindingdb", "source_release", "public_snapshot", "source_release::bindingdb::public_snapshot", "source_release", confidence="provenance_only", provenance="Output/audits/source_release_manifest.tsv")
    if binding.is_file():
        with binding.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                provenance = f"bindingdb::{row.get('edge_uid') or 'edge'}"
                add("bindingdb", "compound", row.get("compound_uid"), row.get("compound_uid"), "BindingDB_compound_uid", confidence="exact", provenance=provenance)
                add("bindingdb", "protein", row.get("target_uid"), row.get("target_uid"), "BindingDB_target_uid", confidence="exact", provenance=provenance)
                add("bindingdb", "assay", row.get("affinity_type"), f"bindingdb_assay:{row.get('affinity_type')}", "BindingDB_affinity_type", confidence="source_native", provenance=provenance)
                unit = row.get("affinity_unit")
                canonical, formula = UNIT_MAP.get(str(unit), ("", "unresolved"))
                add("bindingdb", "unit", unit, canonical, "UCUM", raw_unit=unit, canonical_unit=canonical, formula=formula, confidence="exact" if canonical else "unresolved", provenance=provenance)
                for disease in str(row.get("disease_context") or "").split("|"):
                    if disease:
                        add("bindingdb", "disease", disease, f"disease:{_canonical_text(disease)}", "MONDO_or_source_disease", confidence="source_native", provenance=provenance)
                note = _normalise_token(row.get("note"))
                if note:
                    add("bindingdb", "publication", note, f"bindingdb_publication:{hashlib.sha256(note.encode()).hexdigest()[:16]}", "BindingDB_provenance_note", confidence="provenance_only", provenance=provenance)

    entity_counts: dict[str, int] = {entity: 0 for entity in required}
    status_counts: dict[str, int] = {}
    for row in table_rows:
        entity_counts[str(row["entity_type"])] = entity_counts.get(str(row["entity_type"]), 0) + 1
        status_counts[str(row["mapping_status"])] = status_counts.get(str(row["mapping_status"]), 0) + 1
    source_entity_counts: dict[str, dict[str, int]] = {}
    for row in table_rows:
        source_entity_counts.setdefault(str(row["source_id"]), {})[str(row["entity_type"])] = source_entity_counts.setdefault(str(row["source_id"]), {}).get(str(row["entity_type"]), 0) + 1
    role_reports: dict[str, dict[str, object]] = {}
    for source_id, axes in PHASE1A_ROLE_REQUIREMENTS.items():
        counts = source_entity_counts.get(source_id, {})
        missing_axes = []
        for axis in axes:
            entity_axis = "unit" if axis.endswith("_unit") else axis
            if counts.get(entity_axis, 0) == 0:
                missing_axes.append(axis)
        role_reports[source_id] = {
            "decision": PHASE1A_DECISIONS[source_id][0],
            "required_axes": list(axes),
            "mapped_entity_counts": counts,
            "missing_axes": missing_axes,
            "mapping_version": "public_foundation_corpus_v1_crosswalk_2026-07-17",
            "status": "passed" if not missing_axes else "not_passed",
            "quarantined_values_are_excluded": True,
        }
    write_tsv(table_path, list(table_rows[0].keys()) if table_rows else ["source_id", "entity_type"], table_rows)
    unresolved_required = [entity for entity in required if entity_counts.get(entity, 0) == 0]
    return {
        "status": "passed" if not unresolved_required and ambiguities == 0 and all(report["status"] == "passed" for report in role_reports.values()) else "not_passed",
        "record_count": len(table_rows),
        "entity_type_counts": entity_counts,
        "required_entity_types": list(required),
        "unit_ontologies": ["UCUM", "UO"],
        "ontology_versions": {"gene": "HGNC_entity_catalog_snapshot_2026-07-17", "cell_type": "CL_2026-07-17", "brain_region": "UBERON_2026-07-17", "species": "NCBITaxon_2026-07-17", "unit": "UCUM_UO_mapping_v1"},
        "mapping_version": "public_foundation_corpus_v1_crosswalk_2026-07-17",
        "critical_mapping_conflict_count": 0,
        "ambiguity_count": ambiguities,
        "quarantined_mapping_count": quarantined,
        "mapping_status_counts": status_counts,
        "source_role_requirements": PHASE1A_ROLE_REQUIREMENTS,
        "source_role_reports": role_reports,
        "unresolved_required_mapping_count": len(unresolved_required),
        "unresolved_required_entities": unresolved_required,
        "crosswalk_table_binding": _binding(repo_root, relative_repo_path(repo_root, table_path)),
        "blocker_codes": [] if not unresolved_required and ambiguities == 0 and all(report["status"] == "passed" for report in role_reports.values()) else ["required_entity_crosswalk_incomplete"],
    }


def build_actions(
    repo_root: Path,
    *,
    validated_observation_blocks: list[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    paths = {
        "regimen_catalog": "Input/processed/action_ready/regimen_catalog.tsv",
        "drug_program_catalog": "Input/processed/action_ready/drug_program_catalog.tsv",
        "compound_pk_prior_catalog": "Input/processed/pkpd_ready/compound_pk_prior_catalog.tsv",
    }
    catalogs = []
    total = 0
    for asset_id, relative in paths.items():
        path = safe_repo_path(repo_root, relative)
        count = count_tsv_rows(path)
        total += count
        catalogs.append({"asset_id": asset_id, "path": relative, "record_count": count, "sha256": sha256_file(path) if path.is_file() else None, "disposition": "migration_input_only"})
    qualified_catalogs: list[dict[str, object]] = []
    qualified_count = 0
    config = load_config(safe_repo_path(repo_root, "Libs/configs/config_stack.yaml"))
    qualified_specs = (
        (
            "phase1e_perturbation_materialization",
            "clue_touchstone_v1_1_npc_neu_perturbation_phase1e_v1",
            "Libs.scripts.validators.validate_phase1e_clue_npc_neu_mainline_block",
            "phase1e_clue_npc_neu_qualified_actions",
            False,
        ),
        (
            "phase1f_perturbation_pair_materialization",
            "pfc_arrayexpress_e_mtab_6772_perturbation_phase1f_v1",
            "Libs.scripts.validators.validate_phase1f_perturbation_pair_block",
            "phase1f_arrayexpress_e_mtab_6772_qualified_actions",
            False,
        ),
    )
    for policy_key, canonical_uid, validator_module, asset_id, redistribution_allowed in qualified_specs:
        policy = config["public_foundation_corpus"][policy_key]
        admission_path = safe_repo_path(repo_root, str(policy["admission_audit_path"]))
        if not admission_path.is_file():
            continue
        validated_block: Mapping[str, object] | None = None
        if validated_observation_blocks is None:
            module = __import__(validator_module, fromlist=["validate"])
            report = getattr(module, "validate")(repo_root)
            if report.get("status") != "passed":
                raise ValueError(
                    "Qualified perturbation action subset failed validation: "
                    + json.dumps(report.get("blockers", []), sort_keys=True)
                )
        else:
            matching = [
                row for row in validated_observation_blocks
                if row.get("canonical_uid") == canonical_uid
                and row.get("observation_eligible") is True
                and row.get("role") == "qualified_for_mainline"
            ]
            if len(matching) != 1:
                raise ValueError("Qualified perturbation action subset lacks exactly one validator-gated observation block")
            validated_block = matching[0]
        admission = _read_json(repo_root, str(policy["admission_audit_path"]))
        manifest_binding = admission.get("canonical_block_manifest")
        if not isinstance(manifest_binding, Mapping):
            raise ValueError("Perturbation admission lacks a canonical manifest binding")
        manifest_path = safe_repo_path(repo_root, str(manifest_binding.get("path", "")))
        if not manifest_path.is_file() or sha256_file(manifest_path) != manifest_binding.get("sha256"):
            raise ValueError("Perturbation admission manifest binding has drifted")
        manifest = _read_json(repo_root, relative_repo_path(repo_root, manifest_path))
        if validated_block is not None and (
            validated_block.get("manifest_path") != relative_repo_path(repo_root, manifest_path)
            or validated_block.get("manifest_sha256") != sha256_file(manifest_path)
        ):
            raise ValueError("Validator-gated perturbation observation block differs from the action manifest")
        action_path = manifest_path.parent / str(manifest["action_catalog_path"])
        action_frame = pd.read_parquet(action_path)
        catalog_count = int(len(action_frame))
        if catalog_count <= 0 or catalog_count != int(manifest["action_count"]):
            raise ValueError("Qualified perturbation action catalog is empty or count-mismatched")
        required_action_fields = {
            "action_uid", "agent_uid", "target_set_or_explicit_unknown",
            "physical_dose", "dose_unit", "duration", "duration_unit", "schedule",
            "route_or_not_applicable", "cell_or_tissue_context", "assay_uid",
            "publication_uid", "accession_uid",
            "exposure_program_ref_or_explicit_unavailable",
            "occupancy_program_ref_or_explicit_unavailable",
        }
        if not required_action_fields.issubset(action_frame.columns):
            raise ValueError("Qualified perturbation action catalog lacks required physical/provenance fields")
        if action_frame[list(required_action_fields)].astype(str).apply(lambda column: column.str.strip().eq("").any()).any():
            raise ValueError("Qualified perturbation action catalog contains blank required values")
        qualified_count += catalog_count
        _, action_bytes, action_hash = hash_path(action_path)
        qualified_catalogs.append(
            {
                "asset_id": asset_id,
                "path": relative_repo_path(repo_root, action_path),
                "record_count": catalog_count,
                "bytes": action_bytes,
                "sha256": action_hash,
                "source_id": manifest["source_id"],
                "source_accession": manifest["source_accession"],
                "canonical_observation_uid": manifest["canonical_uid"],
                "role": "qualified_for_mainline",
                "action_linkage_complete": True,
                "redistribution_allowed": redistribution_allowed,
            }
        )
    return {
        "status": "passed" if qualified_count > 0 else "not_passed",
        "record_count": total + qualified_count,
        "required_fields": list(ACTION_FIELDS),
        "migration_input_record_count": total,
        "qualified_record_count": qualified_count,
        "catalogs": catalogs,
        "qualified_catalogs": qualified_catalogs,
        "blocker_codes": [] if qualified_count > 0 else ["physical_action_contract_incomplete", "publication_accession_missing", "exposure_occupancy_reference_missing"],
    }


def build_tools() -> dict[str, object]:
    rows = [
        {"tool_family": "neuronsifter_mainline", "tool_name": "neuronsifter_world_action", "tool_version": "repository_configured", "official_reference": "Libs/models/modules/neuronsifter_world_action.py", "parser_or_adapter_version": "native_runtime", "smoke_status": "registered"},
        {"tool_family": "neural_operator_baseline", "tool_name": "transolver_microenvironment_operator", "tool_version": "repository_configured", "official_reference": "Libs/models/modules/transolver_operator.py", "parser_or_adapter_version": "baseline_adapter", "smoke_status": "registered_baseline_only"},
        {"tool_family": "pkpd_teacher_adapter", "tool_name": "osp_pbpk_teacher", "tool_version": "registry_configured", "official_reference": "Libs/configs/runtime/osp_pksim_environment.yaml", "parser_or_adapter_version": "registry_adapter", "smoke_status": "registry_only"},
        {"tool_family": "electrophysiology_teacher_adapter", "tool_name": "neuron_rxd_teacher", "tool_version": "registry_configured", "official_reference": "Libs/scripts/teacher_generation/prepare_neuron_rxd_public_sources.py", "parser_or_adapter_version": "registry_adapter", "smoke_status": "registry_only"},
        {"tool_family": "external_signature_adapter", "tool_name": "clue_lincs_adapter", "tool_version": "source_snapshot", "official_reference": "Input/raw/clue_lincs/manifests/source_manifest.json", "parser_or_adapter_version": "pending_cns_qualification", "smoke_status": "registry_only"},
        {"tool_family": "target_evidence_adapter", "tool_name": "bindingdb_adapter", "tool_version": "source_snapshot", "official_reference": "Input/raw/bindingdb/manifests/source_manifest.json", "parser_or_adapter_version": "pending_public_recertification", "smoke_status": "registry_only"},
    ]
    return {
        "status": "passed",
        "record_count": len(rows),
        "tools": rows,
        "generated_trajectories_count_as_observations": False,
        "blocker_codes": [],
    }


SPLIT_FIELDS = [
    "canonical_uid", "content_hash", "family_uid", "source_id", "source_accession",
    "publication_accession_group", "mirror_group", "biological_unit_uid",
    "regimen_uid", "target_program_proxy", "source_holdout_role", "split_role",
]


def _split_units(repo_root: Path, blocks: list[Mapping[str, object]]) -> list[dict[str, str]]:
    units: list[dict[str, str]] = []
    for block in blocks:
        if block.get("observation_eligible") is not True:
            continue
        manifest_path = safe_repo_path(repo_root, str(block["manifest_path"]))
        manifest = _read_json(repo_root, relative_repo_path(repo_root, manifest_path))
        metadata = pd.read_parquet(manifest_path.parent / str(manifest["sample_metadata_path"]))
        family_id = str(block["family_id"])
        if family_id == "cns_cell_type_region_atlas_transcriptomics":
            unit_column = "donor_id" if "donor_id" in metadata else "donor_label"
        elif "biological_unit_uid" in metadata:
            unit_column = "biological_unit_uid"
        else:
            unit_column = "biological_unit_id"
        keep_columns = [unit_column]
        for field in ("action_uid", "compound_uid"):
            if field in metadata:
                keep_columns.append(field)
        unit_frame = metadata[keep_columns].drop_duplicates().copy()
        if unit_frame[unit_column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Blank frozen biological unit in {manifest_path}")
        for row in unit_frame.to_dict("records"):
            biological_uid = str(row[unit_column])
            action_uid = str(row.get("action_uid") or f"not_applicable:{family_id}")
            compound_uid = str(row.get("compound_uid") or f"not_applicable:{family_id}")
            canonical_uid = f"pfc_unit:{canonical_hash([manifest['canonical_uid'], biological_uid])[:24]}"
            units.append({
                "canonical_uid": canonical_uid,
                "content_hash": canonical_hash([manifest["block_sha256"], biological_uid]),
                "family_uid": family_id,
                "source_id": str(block["source_id"]),
                "source_accession": str(block["source_accession"]),
                "publication_accession_group": str(block["publication_accession_group"]),
                "mirror_group": str(block["mirror_group"]),
                "biological_unit_uid": biological_uid,
                "regimen_uid": action_uid,
                "target_program_proxy": f"explicit_unknown_target_agent_proxy:{compound_uid}",
                "source_holdout_role": str(block["source_id"]),
                "split_role": "unassigned",
            })
    if len({row["canonical_uid"] for row in units}) != len(units):
        raise ValueError("Frozen query domain contains duplicate canonical unit UIDs")
    return sorted(units, key=lambda row: row["canonical_uid"])


def _validation_groups(rows: list[dict[str, str]], seed: str, fraction: float) -> set[str]:
    groups = sorted({row["biological_unit_uid"] for row in rows})
    if len(groups) < 2:
        return set()
    count = min(len(groups) - 1, max(1, int(round(len(groups) * fraction))))
    return set(sorted(groups, key=lambda value: canonical_hash([seed, "validation", value]))[:count])


def _global_source_holdout_roles(units: list[dict[str, str]], seed: str) -> dict[str, str]:
    graph: dict[str, set[str]] = {}
    for family_id in CORE_FAMILIES:
        sources = sorted({row["source_id"] for row in units if row["family_uid"] == family_id})
        if len(sources) != 2:
            raise ValueError(f"Frozen V1 source holdout requires exactly two sources for {family_id}")
        left, right = sources
        graph.setdefault(left, set()).add(right)
        graph.setdefault(right, set()).add(left)
    colors: dict[str, int] = {}
    for component_root in sorted(graph, key=lambda value: canonical_hash([seed, "source_component", value])):
        if component_root in colors:
            continue
        colors[component_root] = 1
        queue = [component_root]
        while queue:
            source = queue.pop(0)
            for neighbor in sorted(graph[source]):
                expected = 1 - colors[source]
                if neighbor in colors and colors[neighbor] != expected:
                    raise ValueError("Source independence graph is not bipartite; strict source holdout cannot be frozen")
                if neighbor not in colors:
                    colors[neighbor] = expected
                    queue.append(neighbor)
    roles = {source: "test" if color == 1 else "development" for source, color in colors.items()}
    for family_id in CORE_FAMILIES:
        family_roles = {roles[row["source_id"]] for row in units if row["family_uid"] == family_id}
        if family_roles != {"development", "test"}:
            raise ValueError(f"Global source holdout does not cross both roles for {family_id}")
    return roles


def _write_split_view(
    repo_root: Path,
    split_root: Path,
    *,
    split_uid: str,
    split_type: str,
    rows: list[dict[str, str]],
    applicable_keys: list[str],
    seed: str,
    validation_fraction: float,
) -> dict[str, object]:
    role_rows = {role: [] for role in ("train", "validation", "test")}
    for row in rows:
        role = str(row["split_role"])
        if role not in role_rows:
            raise ValueError(f"Invalid split role {role!r} in {split_uid}")
        role_rows[role].append(dict(row))
    if not role_rows["train"] or not role_rows["test"]:
        raise ValueError(f"Frozen split {split_uid} lacks train or test units")
    role_bindings: dict[str, dict[str, object]] = {}
    for role, assigned in role_rows.items():
        path = split_root / f"{split_uid}.{role}.tsv"
        write_tsv(path, SPLIT_FIELDS, assigned)
        role_bindings[role] = {
            "path": relative_repo_path(repo_root, path),
            "record_count": len(assigned),
            "sha256": sha256_file(path),
        }
    payload: dict[str, object] = {
        "schema_version": "neuronsifter-public-foundation-corpus-split-v1",
        "split_uid": split_uid,
        "split_type": split_type,
        "corpus_hash": canonical_hash(sorted(row["content_hash"] for row in rows)),
        "group_axes": applicable_keys,
        "train_uids_uri": role_bindings["train"]["path"],
        "validation_uids_uri": role_bindings["validation"]["path"],
        "test_uids_uri": role_bindings["test"]["path"],
        "holdout_uids_uri": role_bindings["test"]["path"],
        "manifest_hash": "",
        "leakage_audit_uri": relative_repo_path(repo_root, LEAKAGE_PATH),
        "frozen": True,
        "deterministic_seed": seed,
        "validation_group_fraction": validation_fraction,
        "role_bindings": role_bindings,
        "applicable_zero_overlap_keys": applicable_keys,
    }
    payload["manifest_hash"] = canonical_hash({key: value for key, value in payload.items() if key != "manifest_hash"})
    return payload


def build_splits(
    repo_root: Path,
    blocks: list[Mapping[str, object]],
    split_policy: Mapping[str, object],
    *,
    split_root: Path,
    primary_manifest_path: Path,
) -> dict[str, object]:
    units = _split_units(repo_root, blocks)
    if not units:
        return {"status": "not_passed", "record_count": 0, "query_domain_status": "not_available", "split_manifest_status": "not_frozen", "frozen": False, "manifests": [], "blocker_codes": ["no_qualified_observation_domain"]}
    seed = str(split_policy["deterministic_seed"])
    validation_fraction = float(split_policy["validation_group_fraction"])
    split_root.mkdir(parents=True, exist_ok=True)
    query_path = split_root / "query_domain.tsv"
    write_tsv(query_path, SPLIT_FIELDS, units)

    source_roles = _global_source_holdout_roles(units, seed)
    development_units = [row for row in units if source_roles[row["source_id"]] == "development"]
    validation = _validation_groups(development_units, seed, validation_fraction)
    primary_rows: list[dict[str, str]] = []
    for family_id in CORE_FAMILIES:
        family_rows = [row for row in units if row["family_uid"] == family_id]
        for row in family_rows:
            assigned = dict(row)
            assigned["split_role"] = "test" if source_roles[row["source_id"]] == "test" else ("validation" if row["biological_unit_uid"] in validation else "train")
            primary_rows.append(assigned)
    manifests = [
        _write_split_view(repo_root, split_root, split_uid="source_publication_holdout", split_type="source_holdout", rows=primary_rows, applicable_keys=["canonical_uid", "content_hash", "publication_accession_group", "source_holdout_role"], seed=seed, validation_fraction=validation_fraction)
    ]
    for heldout_family in CORE_FAMILIES:
        candidates = [row for row in units if row["family_uid"] != heldout_family]
        validation = _validation_groups(candidates, seed + heldout_family, validation_fraction)
        assigned_rows = []
        for row in units:
            assigned = dict(row)
            assigned["split_role"] = "test" if row["family_uid"] == heldout_family else ("validation" if row["biological_unit_uid"] in validation else "train")
            assigned_rows.append(assigned)
        manifests.append(_write_split_view(repo_root, split_root, split_uid=f"family_holdout_{heldout_family}", split_type="family_holdout", rows=assigned_rows, applicable_keys=["canonical_uid", "content_hash", "family_uid"], seed=seed, validation_fraction=validation_fraction))

    perturbation = [row for row in units if row["family_uid"] == "compound_dose_time_perturbation_transcriptomics"]
    for split_type, group_key in (("regimen_holdout", "regimen_uid"), ("target_program_holdout", "target_program_proxy")):
        groups = sorted({row[group_key] for row in perturbation}, key=lambda value: canonical_hash([seed, split_type, value]))
        test_count = min(len(groups) - 2, max(1, int(round(len(groups) * 0.2))))
        test_groups = set(groups[:test_count])
        remaining = [row for row in perturbation if row[group_key] not in test_groups]
        validation_groups = set(sorted({row[group_key] for row in remaining}, key=lambda value: canonical_hash([seed, split_type, "validation", value]))[:max(1, int(round(len(set(row[group_key] for row in remaining))) * validation_fraction))])
        assigned_rows = []
        for row in perturbation:
            assigned = dict(row)
            assigned["split_role"] = "test" if row[group_key] in test_groups else ("validation" if row[group_key] in validation_groups else "train")
            assigned_rows.append(assigned)
        manifests.append(_write_split_view(repo_root, split_root, split_uid=split_type, split_type=split_type, rows=assigned_rows, applicable_keys=["canonical_uid", "content_hash", group_key], seed=seed, validation_fraction=validation_fraction))

    primary_manifest = manifests[0]
    write_json(primary_manifest_path, primary_manifest)
    return {
        "status": "passed",
        "record_count": len(units),
        "qualified_observation_block_count": sum(row.get("observation_eligible") is True for row in blocks),
        "query_domain_status": "frozen",
        "split_manifest_status": "frozen",
        "frozen": True,
        "split_seed": seed,
        "query_domain": {"path": relative_repo_path(repo_root, query_path), "record_count": len(units), "sha256": sha256_file(query_path)},
        "nested_split_order": list(split_policy["nested_split_order"]),
        "manifests": manifests,
        "primary_manifest": {"path": relative_repo_path(repo_root, primary_manifest_path), "sha256": sha256_file(primary_manifest_path)},
        "blocker_codes": [],
    }


def build_leakage(repo_root: Path, splits: Mapping[str, object], required_keys: list[str]) -> dict[str, object]:
    manifests = splits.get("manifests", [])
    findings: list[dict[str, object]] = []
    covered: set[str] = set()
    total_overlap = {key: 0 for key in required_keys}
    for manifest in manifests:
        applicable = list(manifest["applicable_zero_overlap_keys"])
        covered.update(applicable)
        role_rows = {
            role: read_tsv(safe_repo_path(repo_root, str(binding["path"])))
            for role, binding in manifest["role_bindings"].items()
        }
        overlaps: dict[str, int] = {}
        for key in applicable:
            total_overlap.setdefault(key, 0)
            role_sets = [{str(row[key]) for row in rows if str(row.get(key) or "")} for rows in role_rows.values()]
            if key in {"source_holdout_role", "publication_accession_group", "family_uid"}:
                # Validation remains inside the development source/publication/family
                # domain. These group axes must be disjoint between development and test.
                overlap_values = (role_sets[0] | role_sets[1]) & role_sets[2]
            else:
                overlap_values = (role_sets[0] & role_sets[1]) | (role_sets[0] & role_sets[2]) | (role_sets[1] & role_sets[2])
            overlaps[key] = len(overlap_values)
            total_overlap[key] += len(overlap_values)
        findings.append({"split_uid": manifest["split_uid"], "applicable_zero_overlap_keys": applicable, "overlap_counts": overlaps, "zero_overlap": not any(overlaps.values())})
    adversarial = {key: adversarial_leakage_test([key]) for key in required_keys}
    uncovered = sorted(set(required_keys) - covered)
    blockers: list[str] = []
    if uncovered:
        blockers.append("required_leakage_key_not_covered")
    if any(total_overlap.values()):
        blockers.append("real_domain_overlap_detected")
    if any(row["status"] != "passed" for row in adversarial.values()):
        blockers.append("adversarial_leakage_detection_failed")
    return {
        "status": "passed" if not blockers else "not_passed",
        "record_count": int(splits.get("record_count", 0)),
        "eligible_observation_count": int(splits.get("record_count", 0)),
        "audit_domain_status": "frozen_real_observation_domain",
        "zero_overlap": not any(total_overlap.values()),
        "required_zero_overlap_keys": required_keys,
        "covered_zero_overlap_keys": sorted(covered),
        "overlap_counts": total_overlap,
        "split_findings": findings,
        "adversarial_injection": {"tested": True, "all_required_keys_detected": all(row["status"] == "passed" for row in adversarial.values()), "by_key": adversarial, "status": "passed" if all(row["status"] == "passed" for row in adversarial.values()) else "not_passed"},
        "blocker_codes": blockers,
    }


def build_qc(blocks: list[Mapping[str, object]], legacy: Mapping[str, object]) -> dict[str, object]:
    grade_counts: dict[str, int] = {}
    for block in blocks:
        grade = str(block.get("qc_grade") or "unknown")
        grade_counts[grade] = grade_counts.get(grade, 0) + 1
    synthetic = legacy.get("synthetic_teachers", {}) if isinstance(legacy, Mapping) else {}
    eligible = [row for row in blocks if row.get("observation_eligible") is True and str(row.get("qc_grade")) in {"A", "B"}]
    return {
        "status": "passed" if eligible else "not_passed",
        "record_count": len(blocks),
        "grade_counts": grade_counts,
        "grade_ab_observation_count": sum(int(row.get("observation_count") or 0) for row in eligible),
        "grade_ab_effective_sample_size": float(sum(float(row.get("effective_sample_size") or 0) for row in eligible)),
        "weak_calibration_count": grade_counts.get("C", 0),
        "reject_count": grade_counts.get("D", 0),
        "synthetic_teacher_observation_count": 0,
        "teacher_catalog_counts": synthetic.get("trajectory_counts", {}),
        "blocker_codes": [] if eligible else ["grade_ab_effective_sample_size_zero", "no_core_observation_blocks"],
    }


def _artifact_record_count(payload: Mapping[str, object]) -> int:
    value = payload.get("record_count")
    return int(value) if isinstance(value, int) else 0


def _ref(repo_root: Path, key: str, path: Path, payload: Mapping[str, object], status: str) -> dict[str, object]:
    return evidence_reference(repo_root, path, schema_version=str(payload.get("schema_version", ARTIFACT_VERSION)), producer=str(payload.get("producer", EVIDENCE_PRODUCER)), record_count=_artifact_record_count(payload), status=status)


def _load_frozen_phase1a_source(
    repo_root: Path,
    audit_root: Path,
) -> dict[str, object]:
    """Copy Phase 1A evidence for isolated builds without ever regenerating it."""

    canonical_root = safe_repo_path(repo_root, "Output/audits")
    for canonical_path in PHASE1A_FROZEN_ARTIFACTS:
        source_path = canonical_root / canonical_path.name
        if not source_path.is_file():
            raise FileNotFoundError(
                "Frozen Phase 1A artifact is required before evidence rebuild: "
                + str(source_path.relative_to(repo_root))
            )
        destination = audit_root / canonical_path.name
        if destination != source_path:
            shutil.copyfile(source_path, destination)
    source = _read_json(repo_root, relative_repo_path(repo_root, canonical_root / SOURCE_PATH.name))
    if source.get("phase1a_status") != "passed":
        raise ValueError("Frozen Phase 1A source qualification is not accepted")
    return source


def build(
    repo_root: Path = REPO_ROOT,
    *,
    run_storage_scan: bool = True,
    audit_root: Path | None = None,
) -> dict[str, object]:
    repo_root = repo_root.resolve()
    audit_root = (
        safe_repo_path(repo_root, "Output/audits")
        if audit_root is None
        else audit_root.resolve()
    )
    canonical_audit_root = safe_repo_path(repo_root, "Output/audits")
    if audit_root != canonical_audit_root and canonical_audit_root not in audit_root.parents:
        raise ValueError("audit_root must be Output/audits or one of its descendants")
    audit_root.mkdir(parents=True, exist_ok=True)
    bundle_path = audit_root / BUNDLE_PATH.name
    legacy_path = audit_root / LEGACY_PATH.name
    legacy_ledger_path = audit_root / "public_foundation_corpus_legacy_recertification.tsv"
    crosswalk_path = audit_root / CROSSWALK_PATH.name
    crosswalk_table_path = audit_root / CROSSWALK_TABLE_PATH.name
    source_path = audit_root / SOURCE_PATH.name
    observation_path = audit_root / OBSERVATION_PATH.name
    action_path = audit_root / ACTION_PATH.name
    tool_path = audit_root / TOOL_PATH.name
    split_path = audit_root / SPLIT_PATH.name
    split_manifest_path = audit_root / SPLIT_MANIFEST_PATH.name
    leakage_path = audit_root / LEAKAGE_PATH.name
    qc_path = audit_root / QC_PATH.name
    storage_path = audit_root / STORAGE_PATH.name
    config = load_config(safe_repo_path(repo_root, "Libs/configs/config_stack.yaml"))
    contract_path = safe_repo_path(repo_root, "Libs/configs/data/public_foundation_corpus.yaml")

    legacy, legacy_rows = build_legacy_recertification(repo_root)
    legacy["record_count"] = len(legacy_rows)
    write_tsv(legacy_ledger_path, LEDGER_FIELDS, legacy_rows)
    legacy["ledger"] = {
        "path": relative_repo_path(repo_root, legacy_ledger_path),
        "row_count": len(legacy_rows),
        "sha256": sha256_file(legacy_ledger_path),
    }
    write_json(legacy_path, legacy)
    source = _load_frozen_phase1a_source(repo_root, audit_root)
    crosswalk = build_crosswalk(repo_root, table_path=crosswalk_table_path)
    observations = build_observation_blocks(repo_root)
    blocks = observations["blocks"]
    actions = build_actions(repo_root, validated_observation_blocks=blocks)
    tools = build_tools()
    split_policy = config["public_foundation_corpus"]["split_policy"]
    configured_split_root = safe_repo_path(repo_root, str(split_policy["split_root"]))
    effective_split_root = (
        configured_split_root
        if audit_root == canonical_audit_root
        else audit_root / "frozen_split_domain"
    )
    splits = build_splits(
        repo_root,
        blocks,
        split_policy,
        split_root=effective_split_root,
        primary_manifest_path=split_manifest_path,
    )
    leakage = build_leakage(
        repo_root,
        splits,
        list(split_policy["required_zero_overlap_keys"]),
    )
    qc = build_qc(blocks, legacy)

    _write_artifact(crosswalk_path, crosswalk)
    _write_artifact(observation_path, observations)
    _write_artifact(action_path, actions)
    _write_artifact(tool_path, tools)
    _write_artifact(split_path, splits)
    _write_artifact(leakage_path, leakage)
    _write_artifact(qc_path, qc)

    storage_policy = config["public_foundation_corpus"]["storage_policy"]
    configured_storage_path = safe_repo_path(
        repo_root, str(storage_policy["storage_ledger_path"])
    )
    if run_storage_scan and configured_storage_path.is_file():
        storage = _read_json(
            repo_root, relative_repo_path(repo_root, configured_storage_path)
        )
    else:
        storage = pending_storage_ledger(
            repo_root,
            roots=storage_policy["scan_root_paths"],
            cap_bytes=int(storage_policy["decimal_byte_cap"]),
            exclude_relative_paths=storage_policy["scan_excluded_paths"],
            previous_confirmed_usage_bytes=int(
                storage_policy["previous_confirmed_usage_bytes"]
            ),
        )
    if not storage:
        storage = pending_storage_ledger(
            repo_root,
            roots=storage_policy["scan_root_paths"],
            cap_bytes=int(storage_policy["decimal_byte_cap"]),
            exclude_relative_paths=storage_policy["scan_excluded_paths"],
            previous_confirmed_usage_bytes=int(
                storage_policy["previous_confirmed_usage_bytes"]
            ),
        )
    if storage_path != configured_storage_path:
        _write_artifact(storage_path, storage)

    isolation = _read_json(repo_root, relative_repo_path(repo_root, ISOLATION_PATH))
    refs = {
        "source_qualification": _ref(repo_root, "source_qualification", source_path, source, source["status"]),
        "legacy_disposition": _ref(
            repo_root,
            "legacy_disposition",
            legacy_path,
            legacy,
            "passed" if legacy.get("audit_execution_status") == "pass" else "not_passed",
        ),
        "entity_unit_crosswalk": _ref(repo_root, "entity_unit_crosswalk", crosswalk_path, crosswalk, crosswalk["status"]),
        "canonical_observation_blocks": _ref(repo_root, "canonical_observation_blocks", observation_path, observations, observations["status"]),
        "structured_action_catalog": _ref(repo_root, "structured_action_catalog", action_path, actions, actions["status"]),
        "tool_model_registry": _ref(repo_root, "tool_model_registry", tool_path, tools, tools["status"]),
        "frozen_splits": _ref(repo_root, "frozen_splits", split_path, splits, splits["status"]),
        "leakage_audit": _ref(repo_root, "leakage_audit", leakage_path, leakage, leakage["status"]),
        "qc_ess_statistics": _ref(repo_root, "qc_ess_statistics", qc_path, qc, qc["status"]),
        "storage_ledger": _ref(
            repo_root,
            "storage_ledger",
            storage_path,
            storage,
            storage_gate_status(storage),
        ),
        "controlled_isolation": evidence_reference(repo_root, ISOLATION_PATH, schema_version=str(isolation.get("schema_version", "unknown")), producer="Libs/scripts/validators/validate_public_mainline_isolation.py", record_count=len(isolation.get("route_findings", [])), status="passed" if isolation.get("status") == "pass" and int(isolation.get("controlled_reachability_count", -1)) == 0 else "not_passed"),
    }

    family_assessments = {
        family_id: family_assessment_from_rows(
            family_id,
            blocks,
            query_domain_status=splits["query_domain_status"],
            nested_split_status="frozen" if splits["frozen"] else "not_frozen",
            leakage_status=leakage["status"],
        )
        for family_id in CORE_FAMILIES
    }
    g1_blockers = sorted({
        "source_qualification_not_passed" if source["status"] != "passed" else "",
        "legacy_disposition_not_passed" if refs["legacy_disposition"]["status"] != "passed" else "",
        "entity_unit_crosswalk_not_passed" if crosswalk["status"] != "passed" else "",
        "storage_cap_pending_revalidation" if storage_gate_status(storage) == "pending_revalidation" else "",
        "storage_cap_exceeded" if storage_gate_status(storage) == "failed_capacity" else "",
        "controlled_reachability_nonzero" if refs["controlled_isolation"]["status"] != "passed" else "",
    } - {""})
    g2_blockers = sorted(
        {
            f"{family_id}:{code}"
            for family_id, assessment in family_assessments.items()
            for code in assessment["blocker_codes"]
        }
    )
    if g1_blockers:
        g2_blockers.append("public_corpus_g1_not_passed")
    g2_blockers.extend(sorted(key + "_not_passed" for key, ref in refs.items() if key not in {"legacy_disposition", "source_qualification", "entity_unit_crosswalk", "controlled_isolation", "leakage_audit", "storage_ledger"} and ref["status"] != "passed"))
    if leakage.get("status") != "passed":
        g2_blockers.append("leakage_audit_not_passed")
    if storage_gate_status(storage) == "pending_revalidation":
        g2_blockers.append("storage_cap_pending_revalidation")
    elif storage_gate_status(storage) == "failed_capacity":
        g2_blockers.append("storage_cap_exceeded")
    g2_blockers = sorted(set(g2_blockers))
    bundle: dict[str, object] = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "corpus_uid": str(config["public_foundation_corpus"]["corpus_version"]),
        "contract": {"ref": "Libs/configs/data/public_foundation_corpus.yaml", "sha256": sha256_file(contract_path)},
        "evidence": refs,
        "family_assessments": family_assessments,
        "paired_benchmark": {"status": "not_available", "provenance_kind": "real_only", "real_pair_count": 0, "cross_modal_prediction_claim_allowed": False, "absent_or_unqualified_claim_policy": "remove_claim"},
        "derived_gates": {
            "iclr_v2.public_corpus.G1": {"gate_id": "iclr_v2.public_corpus.G1", "status": "passed" if not g1_blockers else "not_passed", "fail_closed": True, "blocker_codes": g1_blockers},
            "iclr_v2.public_corpus.G2": {"gate_id": "iclr_v2.public_corpus.G2", "status": "PASS_WITH_SCOPE_LIMITATION" if all(row["status"] == "qualified" for row in family_assessments.values()) and not g2_blockers else "not_passed", "scope": "reduced_transcriptomics_core", "fail_closed": True, "full_multimodal_claim_allowed": False, "blocker_codes": g2_blockers or []},
            "iclr_v2.public_corpus.G2_full_multimodal": {"gate_id": "iclr_v2.public_corpus.G2_full_multimodal", "status": "not_passed", "fail_closed": True, "blocker_codes": ["deferred_modalities_not_qualified", "full_multimodal_gate_locked_closed"]},
        },
    }
    bundle["corpus_hash"] = canonical_hash(bundle)
    write_json(bundle_path, bundle)
    return bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Phase 1 public foundation corpus evidence bundle.")
    parser.add_argument("--output", default=str(BUNDLE_PATH))
    parser.add_argument(
        "--skip-storage",
        action="store_true",
        help="Use an explicit pending ledger for bounded unit tests only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = build(REPO_ROOT, run_storage_scan=not args.skip_storage)
    if args.output != str(BUNDLE_PATH):
        write_json(Path(args.output), bundle)
    print(json.dumps({"corpus_uid": bundle["corpus_uid"], "corpus_hash": bundle["corpus_hash"], "G1": bundle["derived_gates"]["iclr_v2.public_corpus.G1"]["status"], "G2": bundle["derived_gates"]["iclr_v2.public_corpus.G2"]["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
