"""Retrieve official public assets for a configured real-intervention source."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Mapping

import requests


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.public_foundation_evidence import sha256_file  # noqa: E402
from Libs.utils.data.raw_data_registry import (  # noqa: E402
    download_to_file,
    load_config,
    write_json,
)


CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"
DEFAULT_SOURCE = "amyloid_peptide_timecourse"


def _repo_path(value: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    if path != REPO_ROOT and REPO_ROOT not in path.parents:
        raise ValueError(f"Configured path escapes the repository: {value}")
    return path


def _source_contract(source_key: str) -> Mapping[str, object]:
    config = load_config(CONFIG_PATH)
    sources = config["real_intervention_evidence"]["sources"]
    if source_key not in sources:
        raise KeyError(f"Unknown real-intervention source: {source_key}")
    source = sources[source_key]
    if not isinstance(source, Mapping) or not isinstance(source.get("assets"), Mapping):
        raise ValueError(f"Source is not retrievable through this command: {source_key}")
    return source


def _download_with_bounded_retry(url: str, destination: Path, attempts: int = 3) -> None:
    errors: list[str] = []
    for attempt in range(1, attempts + 1):
        try:
            download_to_file(url, destination)
            return
        except (requests.RequestException, OSError) as error:
            errors.append(f"attempt_{attempt}:{type(error).__name__}:{error}")
            part_path = destination.parent / f"{destination.name}.part"
            part_path.unlink(missing_ok=True)
            if attempt < attempts:
                time.sleep(2 ** (attempt - 1))
    raise RuntimeError(f"Official retrieval failed after {attempts} attempts for {url}: {' | '.join(errors)}")


def retrieve(source_key: str = DEFAULT_SOURCE, *, resume_incomplete: bool = False) -> dict[str, object]:
    source = _source_contract(source_key)
    output_root = _repo_path(str(source["official_snapshot_root"]))
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    assets = source["assets"]
    assert isinstance(assets, Mapping)
    for asset_role, raw_contract in assets.items():
        if not isinstance(raw_contract, Mapping):
            raise TypeError(f"Asset contract must be a mapping: {asset_role}")
        filename = str(raw_contract["filename"])
        url = str(raw_contract["url"])
        destination = output_root / filename
        if resume_incomplete and destination.is_file() and destination.stat().st_size > 0:
            retrieval_action = "reused_from_incomplete_same_route_attempt"
        else:
            _download_with_bounded_retry(url, destination)
            retrieval_action = "downloaded_from_official_url"
        if not destination.is_file() or destination.stat().st_size == 0:
            raise RuntimeError(f"Official retrieval produced an empty asset: {url}")
        records.append(
            {
                "asset_role": str(asset_role),
                "path": destination.relative_to(REPO_ROOT).as_posix(),
                "url": url,
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
                "retrieval_action": retrieval_action,
            }
        )
    receipt = {
        "schema_version": "neuronsifter-real-intervention-source-retrieval-v1",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "source_key": source_key,
        "source_id": source["source_id"],
        "source_accession": source["source_accession"],
        "official_study_url": source["official_study_url"],
        "resume_incomplete": resume_incomplete,
        "asset_count": len(records),
        "assets": records,
    }
    write_json(output_root / "retrieval_record.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--resume-incomplete", action="store_true")
    args = parser.parse_args()
    print(json.dumps(retrieve(args.source, resume_incomplete=args.resume_incomplete), sort_keys=True))


if __name__ == "__main__":
    main()
