from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import (
    download_to_file,
    load_config,
    refresh_checksums,
    resolve_repo_path,
)


def snapshot_source(source_config: dict) -> None:
    source_root = resolve_repo_path(source_config["local_root"])
    for snapshot in source_config.get("snapshot_urls", []):
        download_to_file(snapshot["url"], source_root / "downloads" / snapshot["filename"])
    for snapshot in source_config.get("term_urls", []):
        download_to_file(snapshot["url"], source_root / "terms" / snapshot["filename"])
    refresh_checksums(source_root, source_config["checksum_file"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download official public metadata snapshots into Input/raw.")
    parser.add_argument(
        "--config",
        default="Libs/configs/config_stack.yaml",
        help="YAML configuration file for the public raw data registry.",
    )
    parser.add_argument(
        "--sources",
        nargs="*",
        default=None,
        help="Optional source-key subset. Defaults to the v2 public mainline source set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    selected_sources = list(config["source_policy"]["mainline_source_keys"]) if args.sources is None else args.sources
    for source_key in selected_sources:
        source_config = config["sources"][source_key]
        if source_key == "literature_parameter_bank":
            refresh_checksums(resolve_repo_path(source_config["local_root"]), source_config["checksum_file"])
            continue
        snapshot_source(source_config)

    complete_script_logging(pipeline_logger, pipeline_started_at)

if __name__ == "__main__":
    main()
