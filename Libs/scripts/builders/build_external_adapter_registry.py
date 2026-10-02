from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.platform_contracts import stable_uid
from Libs.utils.data.raw_data_registry import ensure_directory, load_config, resolve_repo_path, write_tsv
from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build external evidence-adapter contract catalog.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    return parser.parse_args()


def adapter_rows(config: dict) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for adapter_config in config["validation"]["external_adapter_registry"]["adapters"]:
        source_key = str(adapter_config["source_key"])
        rows.append(
            {
                "adapter_uid": stable_uid("external_adapter", source_key),
                "source_key": source_key,
                "source_name": str(adapter_config["source_name"]),
                "adapter_role": str(adapter_config["adapter_role"]),
                "method_contribution_flag": str(adapter_config["method_contribution_flag"]),
                "raw_root": str(adapter_config["raw_root"]),
                "processed_artifact_path": str(adapter_config["processed_artifact_path"]),
                "required_processed_artifact_flag": str(adapter_config["required_processed_artifact_flag"]),
                "provenance_level": str(adapter_config["provenance_level"]),
                "validation_policy": str(adapter_config["validation_policy"]),
                "public_url": str(adapter_config["public_url"]),
                "note": str(adapter_config["note"]),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    registry_config = config["validation"]["external_adapter_registry"]
    root = resolve_repo_path(str(registry_config["local_root"]))
    ensure_directory(root)
    output_path = root / str(registry_config["relative_paths"]["external_adapter_catalog"])
    rows = adapter_rows(config)
    write_tsv(output_path, config["schemas"]["external_adapter_fields"], rows)
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=output_path,
        external_adapter_row_count=len(rows),
    )


if __name__ == "__main__":
    main()
