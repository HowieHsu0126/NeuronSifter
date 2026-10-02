from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.platform_contracts import intervention_program_rows
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, resolve_repo_path, write_tsv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the intervention-program contract catalog for the AD-first screening twin.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    root = resolve_repo_path(config["twin"]["intervention_program_registry"]["local_root"])
    ensure_directory(root)
    relative_paths = config["twin"]["intervention_program_registry"]["relative_paths"]
    rows = intervention_program_rows(config)
    output_path = root / relative_paths["intervention_program_catalog"]
    write_tsv(output_path, config["schemas"]["intervention_program_fields"], rows)
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=output_path,
        intervention_program_row_count=len(rows),
    )


if __name__ == "__main__":
    main()
