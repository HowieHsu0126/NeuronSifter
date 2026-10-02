"""Retrieve accession-bound public evidence for the E-MTAB-6772 block."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.request


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO_ROOT / "Input/raw/arrayexpress/phase1f/E-MTAB-6772/official_snapshots"

ASSETS = {
    "A-MEXP-932.adf.txt": "https://www.ebi.ac.uk/biostudies/files/A-MEXP-932/A-MEXP-932.adf.txt",
    "pubmed_31267473.json": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&retmode=json&id=31267473",
    "europepmc_31267473.json": "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:31267473&format=json",
    "embl_ebi_terms_of_use.html": "https://www.ebi.ac.uk/about/terms-of-use/",
}


def retrieve(output_root: Path = DEFAULT_ROOT) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for name, url in ASSETS.items():
        request = urllib.request.Request(url, headers={"User-Agent": "NeuronSifter-public-corpus/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
            status = int(response.status)
        if status != 200 or not payload:
            raise RuntimeError(f"Official retrieval failed for {url}: status={status}, bytes={len(payload)}")
        path = output_root / name
        path.write_bytes(payload)
        records.append({"path": path.relative_to(REPO_ROOT).as_posix(), "url": url, "bytes": len(payload)})
    receipt = {
        "schema_version": "neuronsifter-arrayexpress-accession-retrieval-v1",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "source_id": "arrayexpress",
        "source_accession": "E-MTAB-6772",
        "assets": records,
    }
    receipt_path = output_root / "retrieval_record.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    print(json.dumps(retrieve(args.output_root), sort_keys=True))


if __name__ == "__main__":
    main()
