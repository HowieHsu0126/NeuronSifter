from __future__ import annotations

from pathlib import Path

from Libs.utils.data.raw_data_registry import read_tsv, today_iso


REPO_ROOT = Path(__file__).resolve().parents[3]


def relative_repo_path(path: Path | None) -> str:
    if path is None:
        return ""
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def build_receipt_row(
    source_accession: str,
    asset_name: str,
    asset_url: str,
    local_path: Path | None,
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
        "size_bytes": local_path.stat().st_size if local_path is not None and local_path.exists() and local_path.is_file() else 0,
        "note": note,
    }


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
