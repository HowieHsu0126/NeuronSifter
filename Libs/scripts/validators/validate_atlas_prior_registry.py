from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import load_config, read_tsv, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the atlas prior registry outputs.")
    parser.add_argument(
        "--config",
        default="Libs/configs/config_stack.yaml",
        help="YAML configuration file for the raw and processed data registries.",
    )
    return parser.parse_args()


def validate_header(path: Path, expected_fields: list[str]) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
    if header != expected_fields:
        raise ValueError(f"{path} header mismatch: {header}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    atlas_root = resolve_repo_path(config["atlas_prior_registry"]["local_root"])
    relative_paths = config["atlas_prior_registry"]["relative_paths"]
    program_catalog_path = atlas_root / relative_paths["program_catalog"]
    source_catalog_path = atlas_root / relative_paths["source_catalog"]
    for path in [program_catalog_path, source_catalog_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing atlas prior output: {path}")

    validate_header(program_catalog_path, config["schemas"]["atlas_prior_program_fields"])
    validate_header(source_catalog_path, config["schemas"]["atlas_prior_source_fields"])
    program_rows = read_tsv(program_catalog_path)
    source_rows = read_tsv(source_catalog_path)
    pipeline_logger.progress(
        "loaded_atlas_prior_inputs",
        program_row_count=len(program_rows),
        source_row_count=len(source_rows),
    )
    if not program_rows:
        raise ValueError("Atlas prior program catalog must not be empty.")

    valid_source_roots = [
        resolve_repo_path(config["sources"]["allen_brain_cell_atlas"]["local_root"]),
        resolve_repo_path(config["sources"]["biccn_public"]["local_root"]),
    ]
    source_count_by_atlas: dict[str, int] = {}
    for row in source_rows:
        source_count_by_atlas[row["atlas_name"]] = int(row["program_count"])

    for row in program_rows:
        source_file_path = resolve_repo_path(row["source_file"])
        if not source_file_path.exists():
            raise FileNotFoundError(f"Atlas prior source file is missing: {source_file_path}")
        if not any(str(source_file_path).startswith(str(root)) for root in valid_source_roots):
            raise ValueError(f"Atlas prior source file is outside the Allen/BICCN public roots: {source_file_path}")
        if row["atlas_name"] not in source_count_by_atlas:
            raise ValueError(f"Atlas prior row references an unknown atlas source: {row['atlas_name']}")
        if not row["feature_ids"]:
            raise ValueError(f"Atlas prior row is missing feature_ids: {row['atlas_entity_uid']}")
        if row["provenance_level"] not in {"native_atlas", "publication_curated", "inferred"}:
            raise ValueError(f"Unexpected provenance_level in atlas prior row: {row['atlas_entity_uid']}")
        if not row["public_url"]:
            raise ValueError(f"Atlas prior row is missing a public_url: {row['atlas_entity_uid']}")

    for row in source_rows:
        source_file_path = resolve_repo_path(row["source_file"])
        if not source_file_path.exists():
            raise FileNotFoundError(f"Atlas prior source catalog file is missing: {source_file_path}")
        if row["atlas_name"] == "ALLEN_ABC" and int(row["program_count"]) <= 0:
            raise ValueError("ALLEN_ABC atlas prior source must contribute at least one program.")

    pipeline_logger.progress(
        "atlas_prior_validated",
        artifact=program_catalog_path,
        program_row_count=len(program_rows),
        source_row_count=len(source_rows),
    )

    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=program_catalog_path,
        program_row_count=len(program_rows),
        source_row_count=len(source_rows),
    )

if __name__ == "__main__":
    main()
