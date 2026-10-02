from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import load_config, read_tsv, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate alignment-ready study bundles and alignment registry catalogs.")
    parser.add_argument(
        "--config",
        default="Libs/configs/config_stack.yaml",
        help="YAML configuration file for the raw and processed data registries.",
    )
    return parser.parse_args()


def validate_header(path: Path, expected_fields: list[str]) -> None:
    rows = read_tsv(path)
    if not rows and expected_fields:
        with path.open("r", encoding="utf-8") as handle:
            header = handle.readline().rstrip("\n").split("\t")
        if header != expected_fields:
            raise ValueError(f"Header mismatch for {path}: {header}")
        return
    with path.open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
    if header != expected_fields:
        raise ValueError(f"Header mismatch for {path}: {header}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    alignment_root = resolve_repo_path(config["alignment_ready_registry"]["local_root"])
    relative_paths = config["alignment_ready_registry"]["relative_paths"]

    entity_catalog_path = alignment_root / relative_paths["entity_catalog"]
    mapping_catalog_path = alignment_root / relative_paths["study_feature_mapping"]
    matrix_catalog_path = alignment_root / relative_paths["aligned_matrix_catalog"]
    exclusion_catalog_path = alignment_root / relative_paths["alignment_exclusion_catalog"]

    validate_header(entity_catalog_path, config["schemas"]["alignment_entity_fields"])
    validate_header(mapping_catalog_path, config["schemas"]["alignment_feature_mapping_fields"])
    validate_header(matrix_catalog_path, config["schemas"]["alignment_matrix_fields"])
    validate_header(exclusion_catalog_path, config["schemas"]["alignment_exclusion_fields"])

    entity_rows = read_tsv(entity_catalog_path)
    mapping_rows = read_tsv(mapping_catalog_path)
    matrix_rows = read_tsv(matrix_catalog_path)
    exclusion_rows = read_tsv(exclusion_catalog_path)
    pipeline_logger.progress(
        "loaded_alignment_ready_inputs",
        entity_row_count=len(entity_rows),
        mapping_row_count=len(mapping_rows),
        aligned_matrix_row_count=len(matrix_rows),
        exclusion_row_count=len(exclusion_rows),
    )
    entity_ids = {row["canonical_entity_id"] for row in entity_rows}
    matrix_by_study = {row["study_uid"]: row for row in matrix_rows}

    for mapping_row in mapping_rows:
        if mapping_row["canonical_entity_id"] not in entity_ids:
            raise ValueError(f"Mapped canonical entity missing from entity catalog: {mapping_row}")

    expected_entity_index_fields = config["schemas"]["alignment_entity_index_fields"]
    for matrix_row in matrix_rows:
        sample_index_path = resolve_repo_path(matrix_row["sample_index_path"])
        entity_index_path = resolve_repo_path(matrix_row["entity_index_path"])
        raw_matrix_path = resolve_repo_path(matrix_row["matrix_raw_path"])
        norm_matrix_path = resolve_repo_path(matrix_row["matrix_norm_path"])
        manifest_path = resolve_repo_path(matrix_row["manifest_path"])
        for path in [sample_index_path, entity_index_path, raw_matrix_path, norm_matrix_path, manifest_path]:
            if not path.exists():
                raise FileNotFoundError(f"Alignment artifact is missing: {path}")
        sample_index = pd.read_parquet(sample_index_path)
        entity_index = pd.read_parquet(entity_index_path)
        raw_matrix = np.load(raw_matrix_path, allow_pickle=False)
        norm_matrix = np.load(norm_matrix_path, allow_pickle=False)
        if list(entity_index.columns) != expected_entity_index_fields:
            raise ValueError(f"Entity index schema mismatch for {matrix_row['study_uid']}: {list(entity_index.columns)}")
        expected_shape = (sample_index.shape[0], entity_index.shape[0])
        if raw_matrix.shape != expected_shape:
            raise ValueError(f"Aligned raw matrix shape mismatch for {matrix_row['study_uid']}: {raw_matrix.shape} vs {expected_shape}")
        if norm_matrix.shape != expected_shape:
            raise ValueError(f"Aligned normalized matrix shape mismatch for {matrix_row['study_uid']}: {norm_matrix.shape} vs {expected_shape}")
        if int(matrix_row["sample_count"]) != sample_index.shape[0]:
            raise ValueError(f"Sample count mismatch for {matrix_row['study_uid']}")
        if int(matrix_row["aligned_entity_count"]) != entity_index.shape[0]:
            raise ValueError(f"Entity count mismatch for {matrix_row['study_uid']}")

    for exclusion_row in exclusion_rows:
        if exclusion_row["exclusion_scope"] == "study" and exclusion_row["study_uid"] in matrix_by_study:
            raise ValueError(f"Study cannot be both aligned and study-level excluded: {exclusion_row['study_uid']}")

    pipeline_logger.progress(
        "alignment_ready_validated",
        artifact=matrix_catalog_path,
        entity_row_count=len(entity_rows),
        mapping_row_count=len(mapping_rows),
        aligned_matrix_row_count=len(matrix_rows),
        exclusion_row_count=len(exclusion_rows),
    )

    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=matrix_catalog_path,
        entity_row_count=len(entity_rows),
        mapping_row_count=len(mapping_rows),
        aligned_matrix_row_count=len(matrix_rows),
        exclusion_row_count=len(exclusion_rows),
    )

if __name__ == "__main__":
    main()
