from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.platform_contracts import candidate_validation_rows
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, resolve_repo_path, write_tsv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the candidate-validation contract catalog for intervention reranking.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    root = resolve_repo_path(config["validation"]["candidate_validation_registry"]["local_root"])
    ensure_directory(root)
    relative_paths = config["validation"]["candidate_validation_registry"]["relative_paths"]
    rows = candidate_validation_rows(config)
    output_path = root / relative_paths["candidate_validation_catalog"]
    write_tsv(output_path, config["schemas"]["candidate_validation_fields"], rows)
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=output_path,
        candidate_validation_row_count=len(rows),
    )


if __name__ == "__main__":
    main()
