"""Retrieve the bounded official GEO assets for the C1R GSE262992 candidate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.public_foundation_evidence import sha256_file  # noqa: E402
from Libs.utils.data.raw_data_registry import (  # noqa: E402
    download_to_file,
    ensure_directory,
    http_get,
    write_json,
    write_text_atomic,
)


CONFIG_PATH = REPO_ROOT / "Libs/configs/data/public_foundation_corpus_rescue.yaml"
OUTPUT_PATH = REPO_ROOT / "Output/audits/c1_rescue_gse262992_retrieval.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_policy() -> Mapping[str, object]:
    import yaml

    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return payload["public_foundation_corpus_rescue"]


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _binding(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": _relative(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _download_text(url: str, path: Path) -> None:
    response = http_get(url)
    write_text_atomic(path, response.text)


def build(*, force: bool = False) -> dict[str, object]:
    policy = _load_policy()
    candidate = policy["candidates"]["gse262992"]
    assert isinstance(candidate, Mapping)
    raw_root = REPO_ROOT / str(policy["paths"]["raw_root"]) / "geo/GSE262992/official_snapshots"
    ensure_directory(raw_root)
    targets = {
        "raw_archive": (str(candidate["raw_archive_url"]), raw_root / "GSE262992_RAW.tar", False),
        "filelist": (str(candidate["raw_filelist_url"]), raw_root / "filelist.txt", True),
        "series_matrix": (
            str(candidate["series_matrix_url"]),
            raw_root / "GSE262992_series_matrix.txt.gz",
            False,
        ),
        "series_soft": (str(candidate["soft_url"]), raw_root / "GSE262992.soft.txt", True),
        "geo_terms": (str(candidate["terms_url"]), raw_root / "geo_download_terms.html", True),
    }
    for _, (url, path, is_text) in targets.items():
        if force or not path.is_file():
            if is_text:
                _download_text(url, path)
            else:
                download_to_file(url, path)
    archive = targets["raw_archive"][1]
    expected_bytes = int(candidate["expected_raw_archive_bytes"])
    blockers: list[dict[str, object]] = []
    if archive.stat().st_size != expected_bytes:
        blockers.append(
            {
                "code": "gse262992_raw_archive_size_mismatch",
                "expected": expected_bytes,
                "observed": archive.stat().st_size,
            }
        )
    receipt = {
        "schema_version": "neuronsifter-c1-rescue-gse262992-retrieval-v1",
        "status": "passed" if not blockers else "blocked",
        "created_at": _now(),
        "candidate_uid": candidate["candidate_uid"],
        "official_release": candidate["official_release"],
        "source_accession": candidate["source_accession"],
        "access_class": "public_with_attribution",
        "training_role_compatibility": "candidate_pending_versioned_qualification",
        "assets": {
            name: {"official_url": url, **_binding(path)}
            for name, (url, path, _) in targets.items()
        },
        "blockers": blockers,
    }
    write_json(OUTPUT_PATH, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    report = build(force=parse_args().force)
    print(json.dumps(report, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
