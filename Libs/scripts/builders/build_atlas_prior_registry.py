from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, read_tsv, resolve_repo_path, slugify, write_tsv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the public atlas prior registry from Allen ABC and BICCN public assets.")
    parser.add_argument(
        "--config",
        default="Libs/configs/config_stack.yaml",
        help="YAML configuration file for the raw and processed data registries.",
    )
    return parser.parse_args()


def stable_uid(*parts: str) -> str:
    normalized_parts = [part.strip() for part in parts if part and part.strip()]
    joined = "::".join(normalized_parts)
    digest = hashlib.sha1(joined.encode("utf-8")).hexdigest()[:12]
    slug_seed = "_".join(normalized_parts[:4]) if normalized_parts else "atlas"
    return f"{slugify(slug_seed)[:80]}_{digest}"


def relative_repo_path(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def allen_receipt_lookup(source_root: Path) -> tuple[str, str]:
    receipt_rows = read_tsv(source_root / "downloads" / "download_receipts.tsv")
    for row in receipt_rows:
        if row["asset_name"] == "seaad_mapmycells" and row["local_path"].endswith(".h5"):
            return row["asset_url"], row["local_path"]
    raise FileNotFoundError("Allen SEAAD precomputed_stats receipt is missing from the raw registry.")


def allen_gene_symbol_lookup(gene_csv_path: Path) -> dict[str, str]:
    gene_frame = pd.read_csv(gene_csv_path)
    lookup: dict[str, str] = {}
    for row in gene_frame.itertuples(index=False):
        gene_identifier = str(row.gene_identifier)
        gene_symbol = str(row.gene_symbol)
        if gene_identifier and gene_symbol:
            lookup.setdefault(gene_identifier, gene_symbol)
    return lookup


def top_feature_ids(mean_expression: np.ndarray, col_names: list[str], gene_lookup: dict[str, str], top_feature_count: int) -> str:
    candidate_count = min(top_feature_count, mean_expression.shape[0])
    ranked_indices = np.argsort(-mean_expression)[:candidate_count]
    feature_ids: list[str] = []
    seen_feature_ids: set[str] = set()
    for index in ranked_indices:
        gene_identifier = str(col_names[index])
        feature_id = gene_lookup.get(gene_identifier, gene_identifier)
        if not feature_id or feature_id in seen_feature_ids:
            continue
        seen_feature_ids.add(feature_id)
        feature_ids.append(feature_id)
    return "|".join(feature_ids)


def build_allen_program_rows(config: dict) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    atlas_config = config["atlas_prior_registry"]["allen_abc"]
    source_root = resolve_repo_path(config["sources"]["allen_brain_cell_atlas"]["local_root"])
    public_url, source_file = allen_receipt_lookup(source_root)
    source_file_path = resolve_repo_path(source_file)
    release_label = str(atlas_config["release_label"])
    gene_lookup = allen_gene_symbol_lookup(source_root / "downloads" / "releases" / release_label / "WHB-10Xv3" / "gene.csv")

    with h5py.File(source_file_path, "r") as handle:
        cluster_to_row = json.loads(handle["cluster_to_row"][()].decode())
        col_names = json.loads(handle["col_names"][()].decode())
        n_cells = handle["n_cells"][:]
        sum_matrix = handle["sum"][:]
        taxonomy_tree = json.loads(handle["taxonomy_tree"][()].decode())

    supertype_mapper = taxonomy_tree["name_mapper"]["CCN20230505_SUPT"]
    minimum_cell_count = int(atlas_config["minimum_cell_count"])
    top_feature_count = int(atlas_config["top_feature_count"])
    program_rows: list[dict[str, object]] = []
    for cell_type_id, row_index in sorted(cluster_to_row.items(), key=lambda item: item[1]):
        cell_count = int(n_cells[row_index])
        if cell_count < minimum_cell_count:
            continue
        cell_type_payload = supertype_mapper[cell_type_id]
        cell_type_label = str(cell_type_payload.get("alias") or cell_type_payload.get("name") or cell_type_id)
        mean_expression = sum_matrix[row_index] / cell_count
        feature_ids = top_feature_ids(mean_expression, col_names, gene_lookup, top_feature_count)
        if not feature_ids:
            continue
        program_id = f"{cell_type_id}::mean_expression_top_{top_feature_count}"
        program_label = f"{cell_type_label} top mean expression"
        program_rows.append(
            {
                "atlas_entity_uid": stable_uid(
                    atlas_config["atlas_name"],
                    atlas_config["seaad_region_id"],
                    cell_type_id,
                    program_id,
                ),
                "atlas_name": atlas_config["atlas_name"],
                "species": atlas_config["species"],
                "brain_region_id": atlas_config["seaad_region_id"],
                "brain_region_label": atlas_config["seaad_region_label"],
                "cell_type_id": cell_type_id,
                "cell_type_label": cell_type_label,
                "program_id": program_id,
                "program_label": program_label,
                "feature_namespace": "gene_symbol",
                "feature_ids": feature_ids,
                "provenance_level": "native_atlas",
                "public_url": public_url,
                "release_type": "processed",
                "source_file": source_file,
            }
        )

    source_rows = [
        {
            "atlas_name": atlas_config["atlas_name"],
            "source_key": "allen_brain_cell_atlas",
            "release_label": release_label,
            "public_url": public_url,
            "source_file": source_file,
            "program_count": len(program_rows),
            "note": "Program catalog derived from SEAAD precomputed_stats top mean-expression features.",
        }
    ]
    return program_rows, source_rows


def build_biccn_source_rows(config: dict) -> list[dict[str, object]]:
    source_root = resolve_repo_path(config["sources"]["biccn_public"]["local_root"])
    receipt_path = source_root / "downloads" / "download_receipts.tsv"
    if not receipt_path.exists():
        return []
    receipt_rows = read_tsv(receipt_path)
    source_rows: list[dict[str, object]] = []
    for row in receipt_rows:
        if row["asset_name"] != "biccn_data_page":
            continue
        source_rows.append(
            {
                "atlas_name": "BICCN_PUBLIC",
                "source_key": "biccn_public",
                "release_label": config["sources"]["biccn_public"]["release_label"],
                "public_url": row["asset_url"],
                "source_file": row["local_path"],
                "program_count": 0,
                "note": "Public BICCN atlas routing metadata captured; no native program matrix has been materialized into the v2 catalog yet.",
            }
        )
    return source_rows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    program_rows, source_rows = build_allen_program_rows(config)
    source_rows.extend(build_biccn_source_rows(config))
    pipeline_logger.progress(
        "loaded_atlas_prior_inputs",
        allen_program_row_count=len(program_rows),
        source_row_count=len(source_rows),
    )
    program_rows.sort(key=lambda row: (str(row["atlas_name"]), str(row["cell_type_id"])))
    source_rows.sort(key=lambda row: (str(row["atlas_name"]), str(row["source_key"])))

    atlas_root = resolve_repo_path(config["atlas_prior_registry"]["local_root"])
    ensure_directory(atlas_root)
    relative_paths = config["atlas_prior_registry"]["relative_paths"]
    write_tsv(atlas_root / relative_paths["program_catalog"], config["schemas"]["atlas_prior_program_fields"], program_rows)
    write_tsv(atlas_root / relative_paths["source_catalog"], config["schemas"]["atlas_prior_source_fields"], source_rows)
    pipeline_logger.artifact(
        "atlas_prior_registry_written",
        artifact=atlas_root,
        program_row_count=len(program_rows),
        source_row_count=len(source_rows),
    )

    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=atlas_root,
        program_row_count=len(program_rows),
        source_row_count=len(source_rows),
    )

if __name__ == "__main__":
    main()
