from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import load_config, resolve_repo_path, write_tsv  # noqa: E402


SUMMARY_FIELDS = [
    "source_key",
    "local_root",
    "download_file_count",
    "term_file_count",
    "checksum_entries",
    "checksum_bytes",
    "receipt_rows",
    "downloaded_rows",
    "reused_cached_rows",
    "blocked_rows",
    "prepared_rows",
    "not_available_rows",
    "accession_rows",
]


def count_receipt_statuses(receipt_path: Path) -> tuple[int, int, int, int, int, int]:
    if not receipt_path.exists():
        return 0, 0, 0, 0, 0, 0
    with receipt_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return (
        len(rows),
        sum(1 for row in rows if row["status"] == "downloaded"),
        sum(1 for row in rows if row["status"] == "reused_cached"),
        sum(1 for row in rows if row["status"].startswith("blocked_")),
        sum(1 for row in rows if row["status"] == "prepared"),
        sum(1 for row in rows if row["status"] == "not_available"),
    )


def count_checksum_entries(checksum_path: Path) -> tuple[int, int]:
    with checksum_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return len(rows), sum(int(row["size_bytes"]) for row in rows)


def count_accession_rows(accession_path: Path) -> int:
    if not accession_path.exists():
        return 0
    with accession_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the NeuronSifter raw data registry state.")
    parser.add_argument(
        "--config",
        default="Libs/configs/config_stack.yaml",
        help="YAML configuration file for the public raw data registry.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    summary_rows: list[dict[str, object]] = []
    raw_root = resolve_repo_path(config["project"]["raw_root"])

    for source_key, source_config in config["sources"].items():
        source_root = resolve_repo_path(source_config["local_root"])
        public_download_config = config.get("public_downloads", {}).get(source_key, {})
        receipt_path = source_root / public_download_config["receipt_relative_path"] if public_download_config else Path()
        receipt_rows, downloaded_rows, reused_cached_rows, blocked_rows, prepared_rows, not_available_rows = (
            count_receipt_statuses(receipt_path)
            if public_download_config
            else (0, 0, 0, 0, 0, 0)
        )
        checksum_entries, checksum_bytes = count_checksum_entries(source_root / source_config["checksum_file"])
        accession_rows = count_accession_rows(source_root / "accession_freeze" / "accessions.tsv")
        summary_rows.append(
            {
                "source_key": source_key,
                "local_root": source_config["local_root"],
                "download_file_count": sum(1 for path in (source_root / "downloads").rglob("*") if path.is_file()),
                "term_file_count": sum(1 for path in (source_root / "terms").rglob("*") if path.is_file()),
                "checksum_entries": checksum_entries,
                "checksum_bytes": checksum_bytes,
                "receipt_rows": receipt_rows,
                "downloaded_rows": downloaded_rows,
                "reused_cached_rows": reused_cached_rows,
                "blocked_rows": blocked_rows,
                "prepared_rows": prepared_rows,
                "not_available_rows": not_available_rows,
                "accession_rows": accession_rows,
            }
        )

    write_tsv(raw_root / "registry_summary.tsv", SUMMARY_FIELDS, summary_rows)

    complete_script_logging(pipeline_logger, pipeline_started_at)

if __name__ == "__main__":
    main()
