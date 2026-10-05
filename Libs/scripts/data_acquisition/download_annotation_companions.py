from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.exp.logger import bootstrap_script_logging, complete_script_logging
from Libs.utils.data.raw_data_registry import (
    download_to_file,
    ensure_directory,
    load_config,
    read_tsv,
    refresh_checksums,
    resolve_repo_path,
    today_iso,
    write_tsv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download GEO GPL and ArrayExpress ADF companion annotations into Input/raw.")
    parser.add_argument(
        "--config",
        default="Libs/configs/config_stack.yaml",
        help="YAML configuration file for the raw and processed data registries.",
    )
    return parser.parse_args()


def relative_repo_path(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def build_receipt_row(
    source_accession: str,
    asset_name: str,
    asset_url: str,
    local_path: Path,
    status: str,
    note: str,
) -> dict[str, object]:
    return {
        "source_accession": source_accession,
        "asset_name": asset_name,
        "asset_url": asset_url,
        "local_path": relative_repo_path(local_path),
        "download_date": today_iso(),
        "status": status,
        "size_bytes": local_path.stat().st_size if local_path.exists() else 0,
        "note": note,
    }


def gpl_prefix(platform_accession: str) -> str:
    numeric = platform_accession.removeprefix("GPL")
    return f"GPL{numeric[:-3]}nnn" if len(numeric) > 3 else "GPLnnn"


def load_receipt_rows(receipt_path: Path) -> list[dict[str, str]]:
    if not receipt_path.exists():
        return []
    return read_tsv(receipt_path)


def upsert_receipt_rows(
    existing_rows: list[dict[str, str]],
    new_rows: list[dict[str, object]],
    fieldnames: list[str],
) -> list[dict[str, object]]:
    merged: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for row in existing_rows:
        key = (row["source_accession"], row["asset_name"], row["asset_url"], row["local_path"])
        merged[key] = dict(row)
    for row in new_rows:
        key = (
            str(row["source_accession"]),
            str(row["asset_name"]),
            str(row["asset_url"]),
            str(row["local_path"]),
        )
        merged[key] = dict(row)
    ordered_rows = list(merged.values())
    ordered_rows.sort(key=lambda row: (str(row["source_accession"]), str(row["asset_name"]), str(row["local_path"])))
    return [{field: row.get(field, "") for field in fieldnames} for row in ordered_rows]


def geo_platforms_from_model_input(config: dict) -> set[str]:
    model_input_root = resolve_repo_path(config["model_input_registry"]["local_root"])
    study_bundle_rows = read_tsv(model_input_root / config["model_input_registry"]["relative_paths"]["study_bundle_catalog"])
    geo_accessions = sorted(row["source_accession"] for row in study_bundle_rows if row["source_key"] == "geo")
    platform_ids: set[str] = set()
    geo_root = resolve_repo_path(config["sources"]["geo"]["local_root"])
    for accession in geo_accessions:
        family_soft_path = geo_root / "downloads" / "series" / accession / f"{accession}_family.soft.gz"
        if not family_soft_path.exists():
            raise FileNotFoundError(f"Missing GEO family soft for platform discovery: {family_soft_path}")
        with gzip.open(family_soft_path, "rt", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.startswith("!Series_platform_id = "):
                    platform_ids.add(line.split("=", 1)[1].strip())
    return platform_ids


def arrayexpress_designs_from_model_input(config: dict) -> dict[str, set[str]]:
    model_input_root = resolve_repo_path(config["model_input_registry"]["local_root"])
    study_bundle_rows = read_tsv(model_input_root / config["model_input_registry"]["relative_paths"]["study_bundle_catalog"])
    study_accessions = sorted(row["source_accession"] for row in study_bundle_rows if row["source_key"] == "arrayexpress")
    source_root = resolve_repo_path(config["sources"]["arrayexpress"]["local_root"])
    designs_by_accession: dict[str, set[str]] = {}
    for accession in study_accessions:
        sdrf_path = source_root / "downloads" / "studies" / accession / f"{accession}.sdrf.txt"
        if not sdrf_path.exists():
            raise FileNotFoundError(f"Missing ArrayExpress SDRF for design discovery: {sdrf_path}")
        with sdrf_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            designs = {row["Array Design REF"].strip() for row in reader if row.get("Array Design REF", "").strip()}
        designs_by_accession[accession] = designs
    return designs_by_accession


def download_asset(destination: Path, asset_url: str) -> str:
    ensure_directory(destination.parent)
    if destination.exists():
        return "reused_cached"
    download_to_file(asset_url, destination)
    return "downloaded"


def download_geo_platform_companions(config: dict, receipt_rows: list[dict[str, object]]) -> None:
    source_root = resolve_repo_path(config["sources"]["geo"]["local_root"])
    download_config = config["annotation_companion_downloads"]["geo"]
    platform_root = source_root / download_config["platform_relative_dir"]
    gene_reference_root = source_root / download_config["gene_reference_relative_dir"]

    gene_reference_path = gene_reference_root / "Homo_sapiens.gene_info.gz"
    gene_reference_status = download_asset(gene_reference_path, download_config["gene_reference_url"])
    receipt_rows.append(
        build_receipt_row(
            "Homo_sapiens_gene_reference",
            "gene_reference_gene_info",
            download_config["gene_reference_url"],
            gene_reference_path,
            gene_reference_status,
            "NCBI Homo sapiens gene_info companion for symbol-to-entrez normalization",
        )
    )

    for platform_accession in sorted(geo_platforms_from_model_input(config)):
        prefix = gpl_prefix(platform_accession)
        platform_dir = platform_root / platform_accession
        ensure_directory(platform_dir)
        listing_url = download_config["platform_listing_url_template"].format(
            platform_prefix=prefix,
            platform_accession=platform_accession,
        )
        family_url = download_config["platform_family_url_template"].format(
            platform_prefix=prefix,
            platform_accession=platform_accession,
        )
        listing_path = platform_dir / "listing.html"
        family_path = platform_dir / f"{platform_accession}_family.soft.gz"

        listing_status = download_asset(listing_path, listing_url)
        family_status = download_asset(family_path, family_url)
        receipt_rows.append(
            build_receipt_row(
                platform_accession,
                "platform_listing",
                listing_url,
                listing_path,
                listing_status,
                f"GEO platform companion for {platform_accession}",
            )
        )
        receipt_rows.append(
            build_receipt_row(
                platform_accession,
                "platform_family_soft",
                family_url,
                family_path,
                family_status,
                f"GEO platform companion for {platform_accession}",
            )
        )


def download_arrayexpress_platform_companions(config: dict, receipt_rows: list[dict[str, object]]) -> None:
    source_root = resolve_repo_path(config["sources"]["arrayexpress"]["local_root"])
    download_config = config["annotation_companion_downloads"]["arrayexpress"]
    platform_root = source_root / download_config["platform_relative_dir"]
    design_to_geo_platform_alias = download_config["design_to_geo_platform_alias"]
    design_replacement_files = download_config["design_replacement_files"]
    preferred_formats = download_config["geo_platform_preferred_format"]

    for accession, designs in sorted(arrayexpress_designs_from_model_input(config).items()):
        for design_accession in sorted(designs):
            design_dir = platform_root / design_accession
            ensure_directory(design_dir)
            adf_url = download_config["array_design_url_template"].format(array_design_accession=design_accession)
            adf_path = design_dir / f"{design_accession}.adf.txt"
            adf_status = download_asset(adf_path, adf_url)
            receipt_rows.append(
                build_receipt_row(
                    design_accession,
                    "array_design_adf",
                    adf_url,
                    adf_path,
                    adf_status,
                    f"ArrayExpress design companion referenced by {accession}",
                )
            )

            if design_accession in design_replacement_files:
                replacement_config = design_replacement_files[design_accession]
                replacement_path = design_dir / replacement_config["replacement_filename"]
                replacement_status = download_asset(replacement_path, replacement_config["replacement_url"])
                receipt_rows.append(
                    build_receipt_row(
                        str(replacement_config["replacement_accession"]),
                        "array_design_replacement_annotation",
                        str(replacement_config["replacement_url"]),
                        replacement_path,
                        replacement_status,
                        f"Replacement annotation for {design_accession} referenced by {accession}",
                    )
                )

            if design_accession not in design_to_geo_platform_alias:
                continue
            platform_accession = str(design_to_geo_platform_alias[design_accession])
            prefix = gpl_prefix(platform_accession)
            preferred_format = str(preferred_formats.get(platform_accession, preferred_formats["default"]))
            if preferred_format == "miniml_tgz":
                alias_url = download_config["geo_platform_miniml_url_template"].format(
                    platform_prefix=prefix,
                    platform_accession=platform_accession,
                )
                alias_path = design_dir / f"{platform_accession}_family.xml.tgz"
                asset_name = "array_design_geo_platform_alias_miniml"
            else:
                alias_url = download_config["geo_platform_url_template"].format(
                    platform_prefix=prefix,
                    platform_accession=platform_accession,
                )
                alias_path = design_dir / f"{platform_accession}_family.soft.gz"
                asset_name = "array_design_geo_platform_alias"
            alias_status = download_asset(alias_path, alias_url)
            receipt_rows.append(
                build_receipt_row(
                    platform_accession,
                    asset_name,
                    alias_url,
                    alias_path,
                    alias_status,
                    f"GEO platform alias for {design_accession} referenced by {accession}",
                )
            )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline_logger, pipeline_started_at = bootstrap_script_logging(config, args, Path(__file__), sys.argv)
    receipt_fields = config["schemas"]["download_receipt_fields"]

    geo_root = resolve_repo_path(config["sources"]["geo"]["local_root"])
    arrayexpress_root = resolve_repo_path(config["sources"]["arrayexpress"]["local_root"])
    ensure_directory(geo_root / config["annotation_companion_downloads"]["geo"]["platform_relative_dir"])
    ensure_directory(arrayexpress_root / config["annotation_companion_downloads"]["arrayexpress"]["platform_relative_dir"])

    new_geo_receipts: list[dict[str, object]] = []
    new_arrayexpress_receipts: list[dict[str, object]] = []
    download_geo_platform_companions(config, new_geo_receipts)
    download_arrayexpress_platform_companions(config, new_arrayexpress_receipts)

    geo_receipt_path = geo_root / config["annotation_companion_downloads"]["geo"]["receipt_relative_path"]
    arrayexpress_receipt_path = arrayexpress_root / config["annotation_companion_downloads"]["arrayexpress"]["receipt_relative_path"]

    merged_geo_rows = upsert_receipt_rows(load_receipt_rows(geo_receipt_path), new_geo_receipts, receipt_fields)
    merged_arrayexpress_rows = upsert_receipt_rows(
        load_receipt_rows(arrayexpress_receipt_path),
        new_arrayexpress_receipts,
        receipt_fields,
    )

    write_tsv(geo_receipt_path, receipt_fields, merged_geo_rows)
    write_tsv(arrayexpress_receipt_path, receipt_fields, merged_arrayexpress_rows)
    refresh_checksums(geo_root, config["sources"]["geo"]["checksum_file"])
    refresh_checksums(arrayexpress_root, config["sources"]["arrayexpress"]["checksum_file"])
    pipeline_logger.progress(
        "annotation_companions_downloaded",
        artifact=geo_receipt_path,
        geo_companion_count=len(new_geo_receipts),
        arrayexpress_companion_count=len(new_arrayexpress_receipts),
    )

    print(f"geo_companions={len(new_geo_receipts)}")
    print(f"arrayexpress_companions={len(new_arrayexpress_receipts)}")

    complete_script_logging(
        pipeline_logger,
        pipeline_started_at,
        artifact=geo_receipt_path,
        geo_companion_count=len(new_geo_receipts),
        arrayexpress_companion_count=len(new_arrayexpress_receipts),
    )

if __name__ == "__main__":
    main()
