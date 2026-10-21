#!/usr/bin/env python3
"""Retrieve bounded public supplementary tables for Round-19 E2 screening."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import tempfile
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET
import zipfile


EUROPE_PMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
ALLOWED_SUFFIXES = frozenset({".csv", ".tsv", ".xlsx", ".xls"})
LICENSE_PATTERNS = (
    re.compile(r"creativecommons\.org/licenses/by(?:-[a-z]+)*/", re.IGNORECASE),
    re.compile(r"creativecommons\.org/publicdomain/", re.IGNORECASE),
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def fetch(url: str, *, max_bytes: int) -> bytes:
    request = Request(url, headers={"User-Agent": "NeuronSifterRound19/1.0 public-evidence-retrieval"})
    with urlopen(request, timeout=120) as response:
        content_length = int(response.headers.get("Content-Length") or 0)
        if content_length > max_bytes:
            raise ValueError(f"Public source exceeds download budget: {content_length} > {max_bytes}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = response.read(min(1024 * 1024, max_bytes - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"Public source exceeded streaming budget: {total} > {max_bytes}")
            chunks.append(chunk)
    return b"".join(chunks)


def license_urls(xml_bytes: bytes) -> list[str]:
    root = ET.fromstring(xml_bytes)
    urls: set[str] = set()
    for element in root.iter():
        for value in element.attrib.values():
            text = str(value)
            if "creativecommons.org/" in text:
                urls.add(text)
    return sorted(urls)


def allowed_license(urls: list[str]) -> bool:
    return any(pattern.search(url) for url in urls for pattern in LICENSE_PATTERNS)


def safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts and path.suffix.lower() in ALLOWED_SUFFIXES


def retrieve(*, pmcid: str, output_root: Path, max_download_bytes: int, max_table_bytes: int) -> dict[str, object]:
    if not re.fullmatch(r"PMC\d+", pmcid):
        raise ValueError("Only canonical public PMC identifiers are allowed")
    fulltext_url = f"{EUROPE_PMC_BASE}/{pmcid}/fullTextXML"
    supplement_url = f"{EUROPE_PMC_BASE}/{pmcid}/supplementaryFiles"
    xml_bytes = fetch(fulltext_url, max_bytes=min(max_download_bytes, 64 * 1024 * 1024))
    licenses = license_urls(xml_bytes)
    if not licenses or not allowed_license(licenses):
        raise ValueError(f"Public license is not resolved for {pmcid}: {licenses}")

    zip_bytes = fetch(supplement_url, max_bytes=max_download_bytes)
    extracted: list[dict[str, object]] = []
    table_root = output_root / "public_sources" / pmcid / "tables"
    with tempfile.TemporaryDirectory(prefix=f"neuronsifter-round19-{pmcid}-") as temporary_dir:
        archive_path = Path(temporary_dir) / f"{pmcid}_supplementary.zip"
        archive_path.write_bytes(zip_bytes)
        if sha256_file(archive_path) != sha256_bytes(zip_bytes):
            raise RuntimeError("Supplementary archive hash changed after local staging")
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                if not safe_member(member.filename):
                    continue
                if member.file_size > max_table_bytes:
                    raise ValueError(f"Supplementary table exceeds extraction budget: {member.filename}")
                destination = table_root / PurePosixPath(member.filename).name
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                extracted.append(
                    {
                        "filename": destination.name,
                        "path": str(destination),
                        "bytes": destination.stat().st_size,
                        "sha256": sha256_file(destination),
                        "archive_member": member.filename,
                    }
                )
    if not extracted:
        raise ValueError(f"No machine-readable supplementary tables found for {pmcid}")

    manifest = {
        "schema_version": "neuronsifter-round19-public-e2-retrieval-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "retrieved_public_tables",
        "pmcid": pmcid,
        "source_class": "public_supplement",
        "source_identity": f"Europe PMC open-access article {pmcid}",
        "source_version": "europe-pmc-live-public-version",
        "fulltext_url": fulltext_url,
        "supplement_url": supplement_url,
        "publicly_available": True,
        "strict_public_only_flag": True,
        "controlled_access_required": False,
        "controlled_source_resolution_count": 0,
        "credential_required": False,
        "institutional_session_dependency": False,
        "license_or_permitted_use": licenses,
        "parser_version": "neuronsifter-round19-europe-pmc-supplement-v1",
        "code_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "adapter_sha256": sha256_file(Path(__file__).resolve()),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not_slurm"),
            "slurm_job_name": os.environ.get("SLURM_JOB_NAME", "not_slurm"),
            "slurm_node": os.environ.get("SLURMD_NODENAME", platform.node()),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION", "not_slurm"),
        },
        "fulltext_xml_sha256": sha256_bytes(xml_bytes),
        "supplement_bundle_sha256": sha256_bytes(zip_bytes),
        "supplement_bundle_bytes": len(zip_bytes),
        "raw_bundle_persisted": False,
        "table_count": len(extracted),
        "tables": sorted(extracted, key=lambda row: str(row["filename"])),
        "endpoint_admission_performed": False,
        "training_authorized": False,
        "training_started": False,
    }
    write_json(output_root / "public_sources" / pmcid / "retrieval_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pmcid", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-download-mib", type=int, default=512)
    parser.add_argument("--max-table-mib", type=int, default=128)
    args = parser.parse_args()
    manifest = retrieve(
        pmcid=args.pmcid,
        output_root=args.output_root.resolve(),
        max_download_bytes=args.max_download_mib * 1024 * 1024,
        max_table_bytes=args.max_table_mib * 1024 * 1024,
    )
    print(json.dumps({"pmcid": args.pmcid, "status": manifest["status"], "tables": manifest["table_count"], "bundle_bytes": manifest["supplement_bundle_bytes"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
