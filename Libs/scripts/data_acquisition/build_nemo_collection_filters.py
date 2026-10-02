from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.scripts.data_acquisition.reprobe_blocked_public_sources import filtered_collection_row
from Libs.utils.data.raw_data_registry import load_config, refresh_checksums, resolve_repo_path, write_tsv
from Libs.utils.data.raw_receipts import build_receipt_row, load_receipt_rows, upsert_receipt_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build NeMO AD or aging primary and human-brain reference collection tables from discovered public collection metadata.")
    parser.add_argument(
        "--config",
        default="Libs/configs/config_stack.yaml",
        help="YAML configuration file for the raw and processed data registries.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the filtered rows without writing the output TSVs.",
    )
    return parser.parse_args()


def manifest_protocol_types(row: dict[str, object]) -> str:
    manifest_protocols = [value.strip() for value in str(row.get("manifest_protocols", "")).split("|") if value.strip()]
    normalized_protocols: list[str] = []
    seen_protocols: set[str] = set()
    for manifest_protocol in manifest_protocols:
        normalized_protocol = manifest_protocol.lower()
        if normalized_protocol in seen_protocols:
            continue
        seen_protocols.add(normalized_protocol)
        normalized_protocols.append(normalized_protocol)
    return "|".join(normalized_protocols)


def candidate_queue_row(filtered_row: dict[str, object]) -> dict[str, object]:
    return {
        "collection_id": filtered_row["collection_id"],
        "short_name": filtered_row["short_name"],
        "name": filtered_row["name"],
        "description": filtered_row["description"],
        "access": filtered_row["access"],
        "taxa": filtered_row["taxa"],
        "keywords": filtered_row["keywords"],
        "modalities": filtered_row["modalities"],
        "anatomical_regions": filtered_row["anatomical_regions"],
        "collection_type": filtered_row["collection_type"],
        "publication_title": filtered_row["publication_title"],
        "publication_doi": filtered_row["publication_doi"],
        "manifest_count": filtered_row["manifest_count"],
        "manifest_urls": filtered_row["manifest_urls"],
        "manifest_types": filtered_row["manifest_types"],
        "manifest_protocol_types": manifest_protocol_types(filtered_row),
        "processed_manifest_count": filtered_row["processed_manifest_count"],
        "raw_manifest_count": filtered_row["raw_manifest_count"],
        "match_scope": filtered_row["match_scope"],
        "match_fields": filtered_row["match_fields"],
        "keyword_match_terms": filtered_row["keyword_match_terms"],
        "eligibility_bucket": filtered_row["eligibility_bucket"],
        "local_candidate_root": f"Input/raw/nemo_archive/downloads/reprobe/candidates/{str(filtered_row['collection_id']).replace(':', '_')}",
        "note": filtered_row["note"],
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    reprobe_config = config["reprobe_public_sources"]["sources"]["nemo_archive"]
    source_root = resolve_repo_path(config["sources"]["nemo_archive"]["local_root"])
    discovered_path = source_root / reprobe_config["discovered_collections_relative_path"]
    if not discovered_path.exists():
        raise FileNotFoundError(f"Missing discovered collections TSV: {discovered_path}")

    discovered_frame = pd.read_csv(discovered_path, sep="\t")
    discovered_rows = discovered_frame.fillna("").to_dict("records")

    primary_rows = [
        filtered_collection_row(
            row,
            list(reprobe_config["human_taxa_terms"]),
            list(reprobe_config["target_access_values"]),
            list(reprobe_config["primary_target_keywords"]),
            list(reprobe_config["primary_match_fields"]),
            str(reprobe_config["primary_match_scope"]),
        )
        for row in discovered_rows
    ]
    primary_rows = [row for row in primary_rows if row is not None]
    primary_collection_ids = {str(row["collection_id"]) for row in primary_rows}

    reference_rows = [
        filtered_collection_row(
            row,
            list(reprobe_config["human_taxa_terms"]),
            list(reprobe_config["target_access_values"]),
            list(reprobe_config["reference_target_keywords"]),
            list(reprobe_config["reference_match_fields"]),
            str(reprobe_config["reference_match_scope"]),
        )
        for row in discovered_rows
        if str(row["collection_id"]) not in primary_collection_ids
    ]
    reference_rows = [row for row in reference_rows if row is not None]

    primary_rows.sort(key=lambda row: (str(row["access_bucket"]), str(row["collection_id"])))
    reference_rows.sort(key=lambda row: (str(row["access_bucket"]), str(row["collection_id"])))

    primary_candidate_rows = [candidate_queue_row(row) for row in primary_rows if str(row["access_bucket"]) == "open"]
    reference_candidate_rows = [candidate_queue_row(row) for row in reference_rows if str(row["access_bucket"]) == "open"]
    primary_candidate_rows.sort(key=lambda row: (str(row["eligibility_bucket"]), str(row["collection_id"])))
    reference_candidate_rows.sort(key=lambda row: (str(row["eligibility_bucket"]), str(row["collection_id"])))

    if args.dry_run:
        pipeline_logger.progress(
            "nemo_collection_filter_dry_run",
            primary_collection_count=len(primary_rows),
            reference_collection_count=len(reference_rows),
            primary_candidate_count=len(primary_candidate_rows),
            reference_candidate_count=len(reference_candidate_rows),
        )
        print("## ad_aging_primary_collections")
        print(pd.DataFrame(primary_rows).to_string(index=False))
        print("\n## human_brain_reference_collections")
        print(pd.DataFrame(reference_rows).to_string(index=False))
        print("\n## ad_aging_primary_candidate_queue")
        print(pd.DataFrame(primary_candidate_rows).to_string(index=False))
        print("\n## human_brain_reference_candidate_queue")
        print(pd.DataFrame(reference_candidate_rows).to_string(index=False))
        complete_script_logging(
            pipeline_logger,
            pipeline_started_at,
            dry_run=True,
            primary_collection_count=len(primary_rows),
            reference_collection_count=len(reference_rows),
            primary_candidate_count=len(primary_candidate_rows),
            reference_candidate_count=len(reference_candidate_rows),
        )
        return

    primary_collections_path = source_root / reprobe_config["primary_collections_relative_path"]
    reference_collections_path = source_root / reprobe_config["reference_collections_relative_path"]
    primary_candidate_queue_path = source_root / reprobe_config["primary_candidate_queue_relative_path"]
    reference_candidate_queue_path = source_root / reprobe_config["reference_candidate_queue_relative_path"]

    primary_collections_preexisting = primary_collections_path.exists()
    reference_collections_preexisting = reference_collections_path.exists()
    primary_candidate_preexisting = primary_candidate_queue_path.exists()
    reference_candidate_preexisting = reference_candidate_queue_path.exists()

    write_tsv(primary_collections_path, config["schemas"]["nemo_ad_aging_collection_fields"], primary_rows)
    write_tsv(reference_collections_path, config["schemas"]["nemo_ad_aging_collection_fields"], reference_rows)
    write_tsv(primary_candidate_queue_path, config["schemas"]["nemo_candidate_queue_fields"], primary_candidate_rows)
    write_tsv(reference_candidate_queue_path, config["schemas"]["nemo_candidate_queue_fields"], reference_candidate_rows)

    receipt_path = source_root / config["public_downloads"]["nemo_archive"]["receipt_relative_path"]
    existing_receipt_rows = load_receipt_rows(receipt_path)
    new_receipt_rows = [
        build_receipt_row(
            "nemo_archive",
            "reprobe_ad_aging_primary_collections_tsv",
            "https://assets.nemoarchive.org/",
            primary_collections_path,
            "reused_cached" if primary_collections_preexisting else "downloaded",
            f"Filtered {len(primary_rows)} AD or aging primary human collections.",
        ),
        build_receipt_row(
            "nemo_archive",
            "reprobe_human_brain_reference_collections_tsv",
            "https://assets.nemoarchive.org/",
            reference_collections_path,
            "reused_cached" if reference_collections_preexisting else "downloaded",
            f"Filtered {len(reference_rows)} human brain reference collections outside the AD or aging primary scope.",
        ),
        build_receipt_row(
            "nemo_archive",
            "reprobe_ad_aging_primary_candidate_queue_tsv",
            "https://assets.nemoarchive.org/",
            primary_candidate_queue_path,
            "reused_cached" if primary_candidate_preexisting else "downloaded",
            f"Prepared {len(primary_candidate_rows)} open AD or aging primary NeMO candidates.",
        ),
        build_receipt_row(
            "nemo_archive",
            "reprobe_human_brain_reference_candidate_queue_tsv",
            "https://assets.nemoarchive.org/",
            reference_candidate_queue_path,
            "reused_cached" if reference_candidate_preexisting else "downloaded",
            f"Prepared {len(reference_candidate_rows)} open human brain reference NeMO candidates.",
        ),
    ]
    merged_receipt_rows = upsert_receipt_rows(existing_receipt_rows, new_receipt_rows, config["schemas"]["download_receipt_fields"])
    write_tsv(receipt_path, config["schemas"]["download_receipt_fields"], merged_receipt_rows)
    refresh_checksums(source_root, config["sources"]["nemo_archive"]["checksum_file"])

    print(
        {
            "primary_collection_row_count": len(primary_rows),
            "reference_collection_row_count": len(reference_rows),
            "primary_candidate_row_count": len(primary_candidate_rows),
            "reference_candidate_row_count": len(reference_candidate_rows),
            "primary_output_path": str(primary_collections_path.relative_to(REPO_ROOT)),
            "reference_output_path": str(reference_collections_path.relative_to(REPO_ROOT)),
            "primary_candidate_output_path": str(primary_candidate_queue_path.relative_to(REPO_ROOT)),
            "reference_candidate_output_path": str(reference_candidate_queue_path.relative_to(REPO_ROOT)),
        }
    )
    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=primary_candidate_queue_path,
        primary_collection_count=len(primary_rows),
        reference_collection_count=len(reference_rows),
        primary_candidate_count=len(primary_candidate_rows),
        reference_candidate_count=len(reference_candidate_rows),
    )


if __name__ == "__main__":
    main()
