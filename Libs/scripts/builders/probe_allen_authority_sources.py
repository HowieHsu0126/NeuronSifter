"""Capture bounded, hash-addressed metadata evidence from official Allen URLs.

The probe deliberately records transport evidence only.  A successful HTTP
response, a dated URL, or a manifest hash never changes a rights decision.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import requests


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config, write_json


MAX_CAPTURE_BYTES = 262_144
REQUEST_TIMEOUT_SECONDS = 30
REQUEST_HEADERS = {
    "User-Agent": "NeuronSifterAuthorityProbe/2026-08-07",
    "Range": f"bytes=0-{MAX_CAPTURE_BYTES - 1}",
}
HTTP_SESSION = requests.Session()
HTTP_SESSION.trust_env = False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture bounded official Allen metadata evidence without payload retrieval.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--output", default="Output/audits/allen_authority_source_probe.json")
    return parser.parse_args()


def _probe(url: str) -> dict[str, Any]:
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    try:
        with HTTP_SESSION.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS, stream=True) as response:
            chunks: list[bytes] = []
            captured_bytes = 0
            for chunk in response.iter_content(chunk_size=65_536):
                if not chunk:
                    continue
                remaining = MAX_CAPTURE_BYTES - captured_bytes
                if remaining <= 0:
                    break
                chunks.append(chunk[:remaining])
                captured_bytes += min(len(chunk), remaining)
                if captured_bytes >= MAX_CAPTURE_BYTES:
                    break
            captured = b"".join(chunks)
            headers = response.headers
            return {
                "url": url,
                "retrieved_at": timestamp,
                "request_range": REQUEST_HEADERS["Range"],
                "status_code": response.status_code,
                "final_url": response.url,
                "content_type": headers.get("Content-Type"),
                "content_length": headers.get("Content-Length"),
                "etag": headers.get("ETag"),
                "last_modified": headers.get("Last-Modified"),
                "accept_ranges": headers.get("Accept-Ranges"),
                "captured_bytes": len(captured),
                "captured_sha256": hashlib.sha256(captured).hexdigest(),
                "capture_truncated": len(captured) >= MAX_CAPTURE_BYTES,
                "transport_status": "received" if response.ok else "http_error",
            }
    except requests.RequestException as error:
        return {
            "url": url,
            "retrieved_at": timestamp,
            "transport_status": "request_error",
            "error_type": type(error).__name__,
            "error": str(error),
        }


def build(repo_root: Path, output_path: Path) -> dict[str, Any]:
    config = load_config(repo_root / "Libs/configs/config_stack.yaml")
    catalog = config["allen_authority_catalog"]
    families: dict[str, dict[str, str]] = dict(catalog["families"])
    records = []
    for family_uid in sorted(families):
        entry = families[family_uid]
        sources = {
            "product_record": entry["product_record"],
            "dataset_record": entry["dataset_record"],
            "release_locator": entry["release_locator"],
            "citation_record": entry["citation_record"],
        }
        records.append({
            "family_uid": family_uid,
            "display_name": entry["display_name"],
            "sources": {role: _probe(url) for role, url in sources.items()},
        })
    payload = {
        "schema_version": "neuronsifter-allen-authority-source-probe",
        "status": "transport_evidence_only",
        "family_count": len(records),
        "maximum_capture_bytes_per_source": MAX_CAPTURE_BYTES,
        "payload_download_performed": False,
        "rights_or_release_decision_changed": False,
        "records": records,
    }
    write_json(output_path, payload)
    return payload


def main() -> None:
    args = _parse_args()
    payload = build(REPO_ROOT, (REPO_ROOT / args.output).resolve())
    success_count = sum(
        1
        for record in payload["records"]
        for source in record["sources"].values()
        if source["transport_status"] == "received"
    )
    print(json.dumps({"family_count": payload["family_count"], "received_source_count": success_count}, sort_keys=True))


if __name__ == "__main__":
    main()
