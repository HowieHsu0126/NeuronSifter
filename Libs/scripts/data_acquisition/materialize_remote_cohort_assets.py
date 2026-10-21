from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import (
    download_to_file,
    load_config,
    read_tsv,
    refresh_checksums,
    repo_root,
    resolve_repo_path,
    today_iso,
    write_tsv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize remote cohort-ready quantitative assets into Input/raw.")
    parser.add_argument(
        "--config",
        default="Libs/configs/config_stack.yaml",
        help="YAML configuration file for the raw and processed data registries.",
    )
    return parser.parse_args()


def rewrite_remote_url(asset_url: str, rewrite_map: dict[str, str]) -> str:
    for source_prefix, target_prefix in rewrite_map.items():
        if asset_url.startswith(source_prefix):
            return target_prefix + asset_url.removeprefix(source_prefix)
    return asset_url


def destination_from_metadata(asset_row: dict[str, str]) -> Path:
    metadata_path = resolve_repo_path(asset_row["raw_metadata_path"])
    destination_name = Path(urlparse(asset_row["asset_url"]).path).name
    if not destination_name:
        raise ValueError(f"Unable to determine destination filename for asset {asset_row['quantitative_asset_uid']}")
    return metadata_path.parent / destination_name


def upsert_receipt_row(
    receipt_rows: list[dict[str, str]],
    new_row: dict[str, str],
) -> list[dict[str, str]]:
    for index, receipt_row in enumerate(receipt_rows):
        if (
            receipt_row["source_accession"] == new_row["source_accession"]
            and receipt_row["asset_name"] == new_row["asset_name"]
            and receipt_row["local_path"] == new_row["local_path"]
        ):
            receipt_rows[index] = new_row
            return receipt_rows
    receipt_rows.append(new_row)
    return receipt_rows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    processed_root = resolve_repo_path(config["cohort_ready_registry"]["local_root"])
    asset_catalog_path = processed_root / config["cohort_ready_registry"]["relative_paths"]["quantitative_asset_catalog"]
    remote_asset_rows = [
        row
        for row in read_tsv(asset_catalog_path)
        if row["local_availability_flag"] == "no"
    ]
    if not remote_asset_rows:
        complete_script_logging(pipeline_logger, pipeline_started_at, remote_asset_count=0, source_count=0)
        return

    rewrite_map = config["cohort_ready_registry"]["remote_url_rewrites"]
    receipt_fieldnames = config["schemas"]["download_receipt_fields"]
    public_downloads = config["public_downloads"]
    rows_by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for asset_row in remote_asset_rows:
        rows_by_source[asset_row["source_key"]].append(asset_row)

    for source_key, source_asset_rows in rows_by_source.items():
        if source_key not in public_downloads:
            raise ValueError(f"Source {source_key} is missing public_downloads configuration.")
        source_root = resolve_repo_path(config["sources"][source_key]["local_root"])
        receipt_path = source_root / public_downloads[source_key]["receipt_relative_path"]
        receipt_rows = read_tsv(receipt_path)

        for asset_row in source_asset_rows:
            rewritten_url = rewrite_remote_url(asset_row["asset_url"], rewrite_map)
            if not rewritten_url.startswith(("http://", "https://")):
                raise ValueError(f"Unsupported remote asset URL after rewrite: {rewritten_url}")
            destination_path = destination_from_metadata(asset_row)
            relative_destination = destination_path.relative_to(repo_root()).as_posix()
            if destination_path.exists():
                status = "reused_cached"
            else:
                download_to_file(rewritten_url, destination_path)
                status = "downloaded"
            receipt_rows = upsert_receipt_row(
                receipt_rows,
                {
                    "source_accession": asset_row["source_accession"],
                    "asset_name": asset_row["asset_name"],
                    "asset_url": asset_row["asset_url"],
                    "local_path": relative_destination,
                    "download_date": today_iso(),
                    "status": status,
                    "size_bytes": str(destination_path.stat().st_size),
                    "note": f"Materialized from cohort-ready remote asset catalog: {asset_row['note']}",
                },
            )

        receipt_rows.sort(key=lambda row: (row["source_accession"], row["asset_name"], row["local_path"]))
        write_tsv(receipt_path, receipt_fieldnames, receipt_rows)
        refresh_checksums(source_root, config["sources"][source_key]["checksum_file"])
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        remote_asset_count=len(remote_asset_rows),
        source_count=len(rows_by_source),
    )


if __name__ == "__main__":
    main()
