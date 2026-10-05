"""Audit live V1 semantic and action anchors for the C1 rescue decision."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.public_foundation_evidence import sha256_file  # noqa: E402
from Libs.utils.data.raw_data_registry import write_json  # noqa: E402


V1_CORPUS_HASH = "06ae88daf3c44be2c2bb96e51c939554e160b205138ec96f471adb5d82eee791"
DEXAMETHASONE_NAMES = frozenset({"dexamethasone"})
DEXAMETHASONE_INCHI_KEY_PREFIX = "UREBDLICKHMUKA"


def _relative(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _available_columns(path: Path) -> list[str]:
    return list(pq.ParquetFile(path).schema.names)


def _canonical_value_counts(
    path: Path,
    candidates: Iterable[str],
) -> tuple[str, dict[str, int]]:
    columns = _available_columns(path)
    column = next((name for name in candidates if name in columns), "")
    if not column:
        return "", {}
    values = pd.read_parquet(path, columns=[column])[column]
    counts = values.fillna("").astype(str).str.strip().value_counts()
    return column, {
        key: int(value)
        for key, value in sorted(counts.items())
        if key and key.lower() not in {"nan", "none", "<na>"}
    }


def _semantic_metadata(repo_root: Path) -> list[dict[str, object]]:
    roots = [
        repo_root
        / "Input/canonical/public_foundation_corpus_v1/blocks/geo/GSE28146/35dbdac27abdb900/sample_metadata.parquet",
        repo_root
        / "Input/canonical/public_foundation_corpus_v1/blocks/arrayexpress/E-GEOD-37263/eb4f5d16ab060ca8/sample_metadata.parquet",
        repo_root
        / "Input/canonical/public_foundation_corpus_v1/blocks/allen_brain_cell_atlas/WHB-10Xv3-Nonneurons-20240330/0d406b9eb593e33c/cell_metadata.parquet",
        repo_root
        / "Input/canonical/public_foundation_corpus_v1/blocks/cellxgene_census/874e8a85-7fe2-487d-9fef-8fe7f0a17872/b8bc4cab7c8a3e57/cell_metadata.parquet",
    ]
    rows: list[dict[str, object]] = []
    for path in roots:
        if not path.is_file():
            raise FileNotFoundError(path)
        region_column, region_counts = _canonical_value_counts(
            path,
            ("brain_region_ontology_id", "brain_region_canonical", "region_ontology_id"),
        )
        disease_column, disease_counts = _canonical_value_counts(
            path,
            ("disease_state_canonical", "disease_state", "disease"),
        )
        rows.append(
            {
                "path": _relative(path, repo_root),
                "sha256": sha256_file(path),
                "region_column": region_column or "not_available",
                "region_counts": region_counts,
                "disease_column": disease_column or "not_available",
                "disease_counts": disease_counts,
            }
        )
    return rows


def _clue_dexamethasone_anchor(repo_root: Path) -> dict[str, object]:
    block_root = (
        repo_root
        / "Input/canonical/public_foundation_corpus_v1/blocks/clue_lincs/NPC_NEU_TRT_CP/5e51623d38597355"
    )
    action_path = block_root / "actions.parquet"
    pert_path = (
        repo_root
        / "Input/raw/clue_lincs/downloads/builds/TouchstoneV1.1/files/GSE92742_Broad_LINCS_pert_info.txt.gz"
    )
    actions = pd.read_parquet(
        action_path,
        columns=[
            "action_uid",
            "agent_uid",
            "dose_value",
            "dose_unit",
            "duration",
            "duration_unit",
            "cell_or_tissue_context",
            "observation_uid",
        ],
    )
    perturbagens = pd.read_csv(
        pert_path,
        sep="\t",
        compression="gzip",
        usecols=["pert_id", "pert_iname", "inchi_key_prefix", "pubchem_cid"],
        dtype=str,
        low_memory=False,
    )
    perturbagens["agent_uid"] = "clue:" + perturbagens["pert_id"].fillna("")
    joined = actions.merge(perturbagens, on="agent_uid", how="left", validate="many_to_one")
    matched = joined[
        joined["pert_iname"].fillna("").str.casefold().isin(DEXAMETHASONE_NAMES)
    ].copy()
    name_match_records = matched[
        [
            "action_uid",
            "agent_uid",
            "pert_iname",
            "inchi_key_prefix",
            "pubchem_cid",
            "dose_value",
            "dose_unit",
            "duration",
            "duration_unit",
            "cell_or_tissue_context",
            "observation_uid",
        ]
    ].to_dict(orient="records")
    records = [
        row
        for row in name_match_records
        if str(row["inchi_key_prefix"]) == DEXAMETHASONE_INCHI_KEY_PREFIX
    ]
    quarantined = [
        {
            **row,
            "quarantine_reason": "name_match_with_conflicting_chemical_identity",
        }
        for row in name_match_records
        if str(row["inchi_key_prefix"]) != DEXAMETHASONE_INCHI_KEY_PREFIX
    ]
    return {
        "status": "available" if records else "not_available",
        "qualified_action_count": len(records),
        "name_match_action_count": len(name_match_records),
        "identity_conflict_quarantine_count": len(quarantined),
        "canonical_agent_name": "dexamethasone",
        "canonical_inchi_key_prefix": DEXAMETHASONE_INCHI_KEY_PREFIX,
        "expected_external_candidate_dose_value": 1.0,
        "expected_external_candidate_dose_unit": "uM",
        "expected_external_candidate_duration": 72.0,
        "expected_external_candidate_duration_unit": "h",
        "action_path": _relative(action_path, repo_root),
        "action_sha256": sha256_file(action_path),
        "perturbagen_path": _relative(pert_path, repo_root),
        "perturbagen_sha256": sha256_file(pert_path),
        "records": records,
        "identity_conflict_quarantine": quarantined,
    }


def _local_disease_candidate(repo_root: Path) -> dict[str, object]:
    sample_path = (
        repo_root
        / "Input/processed/model_input/studies/arrayexpress/E-GEOD-26972/sample_index.parquet"
    )
    alignment_path = (
        repo_root
        / "Input/processed/alignment_ready/studies/arrayexpress/E-GEOD-26972/sample_index.parquet"
    )
    selected = alignment_path if alignment_path.is_file() else sample_path
    samples = pd.read_parquet(selected)
    return {
        "candidate_uid": "disease_arrayexpress_e_geod_26972",
        "status": "locally_materialized_migration_input",
        "sample_count": len(samples),
        "columns": list(samples.columns),
        "sample_rows": samples.fillna("").astype(str).to_dict(orient="records"),
        "sample_index_path": _relative(selected, repo_root),
        "sample_index_sha256": sha256_file(selected),
    }


def build(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    repo_root = repo_root.resolve()
    bundle_path = repo_root / "Output/audits/public_foundation_corpus_evidence_bundle.json"
    bundle = _load_json(bundle_path)
    observed_hash = str(bundle.get("corpus_hash", ""))
    blockers: list[dict[str, str]] = []
    if observed_hash != V1_CORPUS_HASH:
        blockers.append(
            {
                "code": "c1r_v1_corpus_identity_mismatch",
                "expected": V1_CORPUS_HASH,
                "observed": observed_hash,
            }
        )
    if blockers:
        return {
            "schema_version": "neuronsifter-c1-rescue-candidate-anchor-audit-v1",
            "status": "blocked",
            "run_uid": os.environ.get("C1R_RUN_UID", "local_unassigned"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not_applicable"),
            "v1_corpus_hash": observed_hash,
            "blockers": blockers,
        }

    semantic_rows = _semantic_metadata(repo_root)
    hippocampus_sources = [
        row["path"]
        for row in semantic_rows
        if any(
            value == "UBERON:0002421" or "hippocamp" in value.casefold()
            for value in row["region_counts"]
        )
    ]
    return {
        "schema_version": "neuronsifter-c1-rescue-candidate-anchor-audit-v1",
        "status": "passed",
        "run_uid": os.environ.get("C1R_RUN_UID", "local_unassigned"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not_applicable"),
        "v1_corpus_uid": "public_foundation_corpus_v1",
        "v1_corpus_hash": observed_hash,
        "v1_evidence_bundle_sha256": sha256_file(bundle_path),
        "semantic_metadata": semantic_rows,
        "hippocampus_canonical_anchor_source_count": len(hippocampus_sources),
        "hippocampus_canonical_anchor_sources": hippocampus_sources,
        "qualified_clue_dexamethasone_anchor": _clue_dexamethasone_anchor(repo_root),
        "local_disease_candidate": _local_disease_candidate(repo_root),
        "blockers": [],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build(REPO_ROOT)
    if args.output is not None:
        output = args.output if args.output.is_absolute() else REPO_ROOT / args.output
        write_json(output, report)
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
