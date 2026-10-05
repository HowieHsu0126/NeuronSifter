from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Callable, Iterable


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.source_governance import (
    parse_go_obo,
    parse_hgnc_tsv,
    parse_pubmed_metadata_xml,
    parse_reactome_gmt,
)
from Libs.utils.data.raw_data_registry import write_json


LEGACY_ROOT = REPO_ROOT / "orchestrator/worktrees/round-07-p2-t02-metadata-backbone/Input/raw/p2_materialization_wave/raw"
CAS_ROOT = REPO_ROOT / "Input/raw/public_foundation_corpus_v1/content_sha256"
NORMALIZED_ROOT = REPO_ROOT / "Input/processed/public_foundation_corpus_v1/governance_backbone"
MANIFEST_PATH = REPO_ROOT / "Output/audits/public_foundation_corpus_governance_backbone.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

SOURCE_SPECS: tuple[tuple[str, str, Path, Callable[..., Iterable[dict[str, object]]]], ...] = (
    ("hgnc", "rolling_snapshot_2026-07-14", LEGACY_ROOT / "hgnc/rolling_snapshot_2026-07-14/hgnc_complete_set.txt", parse_hgnc_tsv),
    ("gene_ontology", "go_release_2026-06-19", LEGACY_ROOT / "gene_ontology/go_release_2026-06-19/go-basic.obo", parse_go_obo),
    ("reactome", "97", LEGACY_ROOT / "reactome/97/ReactomePathways.gmt.zip", parse_reactome_gmt),
    ("pubmed", "pubmed26n1533", LEGACY_ROOT / "pubmed/pubmed26n1533/pubmed26n1533.xml.gz", parse_pubmed_metadata_xml),
)


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
            count += 1
    temporary.replace(path)
    return count, sha256_file(path)


def build() -> dict[str, object]:
    source_rows: list[dict[str, object]] = []
    for source_id, release, source_path, parser in SOURCE_SPECS:
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        raw_sha = sha256_file(source_path)
        cas_path = CAS_ROOT / raw_sha[:2] / raw_sha
        cas_path.parent.mkdir(parents=True, exist_ok=True)
        if cas_path.exists() and sha256_file(cas_path) != raw_sha:
            raise RuntimeError(f"content-address collision: {cas_path}")
        if not cas_path.exists():
            temporary = cas_path.with_suffix(".part")
            shutil.copyfile(source_path, temporary)
            if sha256_file(temporary) != raw_sha:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f"migration checksum mismatch: {source_id}")
            temporary.replace(cas_path)
        output = NORMALIZED_ROOT / source_id / release / "records.jsonl"
        count, processed_sha = _write_jsonl(
            output,
            parser(cas_path, source_id=source_id, release_identifier=release),
        )
        source_rows.append(
            {
                "source_id": source_id,
                "release_version": release,
                "legacy_migration_path": source_path.relative_to(REPO_ROOT).as_posix(),
                "content_addressed_path": cas_path.relative_to(REPO_ROOT).as_posix(),
                "raw_bytes": source_path.stat().st_size,
                "raw_sha256": raw_sha,
                "processed_path": output.relative_to(REPO_ROOT).as_posix(),
                "processed_sha256": processed_sha,
                "record_count": count,
                "corpus_role": "governance_input_not_observation",
                "observation_eligible": False,
            }
        )
    return {
        "schema_version": "neuronsifter-public-corpus-governance-backbone-v1",
        "producer": "Libs/scripts/builders/build_public_corpus_governance_backbone.py",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_count": len(source_rows),
        "raw_bytes": sum(int(row["raw_bytes"]) for row in source_rows),
        "record_count": sum(int(row["record_count"]) for row in source_rows),
        "observation_count": 0,
        "sources": source_rows,
        "validation_status": "pass",
        "gate_evidence": {"iclr_v2.public_corpus.G1": False, "iclr_v2.public_corpus.G2": False},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(MANIFEST_PATH))
    args = parser.parse_args()
    report = build()
    write_json(Path(args.output), report)
    print(json.dumps({"source_count": report["source_count"], "record_count": report["record_count"], "observation_count": 0}))


if __name__ == "__main__":
    main()
