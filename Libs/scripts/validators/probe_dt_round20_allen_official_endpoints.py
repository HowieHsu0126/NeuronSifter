#!/usr/bin/env python3
"""Probe official Allen landing pages without retrieving payload data.

The probe records endpoint drift and bounded HTML metadata. It never promotes a
page token or a historical listing hint to an official release UID.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Mapping
from urllib.parse import urljoin, urlparse
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_RELATIVE = Path("Libs/configs/validation/round20_track_contracts.yaml")
SOURCE_CARDS_RELATIVE = Path("Output/audits/round20/historical_inputs/allen_source_cards.json")
SNAPSHOT_RELATIVE = Path("Output/audits/round20/allen_endpoint_snapshots")
MAX_SNAPSHOT_BYTES = 1 * 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 20
USER_AGENT = "NeuronSifter-Round20-Allen-endpoint-probe/1.0"
ALLOWED_HOSTS = {
    "alleninstitute.org",
    "brain-map.org",
    "allensdk.readthedocs.io",
}
RELEASE_TOKEN_RE = re.compile(
    r"(?i)(?:/(?:release|releases|version|versions)/[^/?#\" ]+|[?&](?:release|version)=[^&# ]+|(?:^|[-_/])v\d+(?:\.\d+)*)"
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repository, check=True, capture_output=True, text=True
    ).stdout.strip()


def _host_allowed(host: str | None) -> bool:
    normalized = (host or "").lower().rstrip(".")
    return normalized in ALLOWED_HOSTS or any(
        normalized.endswith("." + suffix) for suffix in ALLOWED_HOSTS
    )


class _MetadataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.links: list[str] = []
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag.lower() == "title":
            self._in_title = True
        if tag.lower() == "a" and values.get("href"):
            self.links.append(str(values["href"]))

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title_parts.append(" ".join(data.split()))


def _extract_html_metadata(payload: bytes, base_url: str) -> dict[str, Any]:
    parser = _MetadataParser()
    parser.feed(payload.decode("utf-8", errors="replace"))
    links: list[str] = []
    for raw_link in parser.links:
        absolute = urljoin(base_url, raw_link)
        parsed = urlparse(absolute)
        if parsed.scheme in {"http", "https"} and _host_allowed(parsed.hostname):
            links.append(absolute)
    links = sorted(set(links))
    candidate_tokens = sorted(
        {
            match.group(0)
            for value in links
            for match in RELEASE_TOKEN_RE.finditer(value)
        }
    )
    return {
        "title": " ".join(parser.title_parts).strip() or None,
        "same_authority_links": links[:200],
        "same_authority_link_count": len(links),
        "release_candidate_tokens": candidate_tokens[:100],
        "release_identity_promoted": False,
    }


def _fetch(url: str) -> dict[str, Any]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not _host_allowed(parsed.hostname):
        return {
            "requested_url": url,
            "status": "blocked_non_official_host",
            "http_status": None,
            "final_url": None,
            "content_type": None,
            "size_bytes": 0,
            "sha256": None,
            "etag": None,
            "last_modified": None,
            "error": "non_official_host",
            "payload": b"",
        }
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html"})
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            final_url = str(response.geturl())
            final_host = urlparse(final_url).hostname
            headers = {str(key).lower(): str(value) for key, value in response.headers.items()}
            payload = response.read(MAX_SNAPSHOT_BYTES + 1)
            if not _host_allowed(final_host):
                return {
                    "requested_url": url,
                    "status": "blocked_redirect_non_official_host",
                    "http_status": int(response.status),
                    "final_url": final_url,
                    "content_type": headers.get("content-type"),
                    "size_bytes": len(payload),
                    "sha256": None,
                    "etag": headers.get("etag"),
                    "last_modified": headers.get("last-modified"),
                    "error": "redirected_to_non_official_host",
                    "payload": b"",
                }
            if len(payload) > MAX_SNAPSHOT_BYTES:
                return {
                    "requested_url": url,
                    "status": "blocked_snapshot_exceeds_bound",
                    "http_status": int(response.status),
                    "final_url": final_url,
                    "content_type": headers.get("content-type"),
                    "size_bytes": len(payload),
                    "sha256": None,
                    "etag": headers.get("etag"),
                    "last_modified": headers.get("last-modified"),
                    "error": "snapshot_exceeds_bound",
                    "payload": b"",
                }
            status = "official_endpoint_observed" if int(response.status) == 200 else "official_endpoint_http_error"
            if urlparse(url).path.rstrip("/") != urlparse(final_url).path.rstrip("/"):
                status = "official_endpoint_drift"
            return {
                "requested_url": url,
                "status": status,
                "http_status": int(response.status),
                "final_url": final_url,
                "content_type": headers.get("content-type"),
                "size_bytes": len(payload),
                "sha256": sha256_bytes(payload),
                "etag": headers.get("etag"),
                "last_modified": headers.get("last-modified"),
                "error": None,
                "payload": payload,
                "retrieval_mode": "network",
            }
    except HTTPError as exc:
        headers = {str(key).lower(): str(value) for key, value in exc.headers.items()}
        return {
            "requested_url": url,
            "status": "official_endpoint_http_error",
            "http_status": int(exc.code),
            "final_url": str(exc.geturl() or url),
            "content_type": headers.get("content-type"),
            "size_bytes": 0,
            "sha256": None,
            "etag": headers.get("etag"),
            "last_modified": headers.get("last-modified"),
            "error": f"HTTPError:{exc.code}",
            "payload": b"",
            "retrieval_mode": "network",
        }
    except Exception as exc:
        return {
            "requested_url": url,
            "status": "official_endpoint_probe_failed",
            "http_status": None,
            "final_url": None,
            "content_type": None,
            "size_bytes": 0,
            "sha256": None,
            "etag": None,
            "last_modified": None,
            "error": f"{type(exc).__name__}:{exc}",
            "payload": b"",
            "retrieval_mode": "network",
        }


def _fetch_snapshot(url: str, path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    if len(payload) > MAX_SNAPSHOT_BYTES:
        raise ValueError(f"offline snapshot exceeds bound: {path}")
    return {
        "requested_url": url,
        "status": "official_endpoint_observed",
        "http_status": 200,
        "final_url": url,
        "content_type": "text/html; offline_snapshot=true",
        "size_bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "etag": None,
        "last_modified": None,
        "error": None,
        "payload": payload,
        "retrieval_mode": "offline_snapshot",
        "snapshot_source_basename": path.name,
        "snapshot_source_sha256": sha256_bytes(payload),
    }


def _family_result(card: Mapping[str, Any], page: Mapping[str, Any]) -> dict[str, Any]:
    receipts = list(card.get("release_listing_receipts") or [])
    release_evidence = [
        {
            "relative_path": item.get("relative_path"),
            "sha256": item.get("sha256"),
            "release_hint": item.get("release_hint"),
            "listed_object_count": item.get("listed_object_count"),
        }
        for item in receipts
    ]
    if receipts:
        release_status = "historical_listing_evidence_present_exact_uid_not_promoted"
    else:
        release_status = "release_identity_unresolved"
    blockers = ["dataset_specific_terms_unresolved", release_status]
    if page["status"] in {
        "official_endpoint_drift",
        "official_endpoint_http_error",
        "official_endpoint_probe_failed",
    }:
        blockers.append(page["status"])
    return {
        "family_uid": str(card["source_uid"]),
        "historical_official_url": card.get("official_url"),
        "endpoint": {key: value for key, value in page.items() if key != "payload"},
        "release_evidence": release_evidence,
        "release_status": release_status,
        "release_identity_promoted": False,
        "rights_status": "blocked_dataset_specific_terms",
        "blockers": sorted(set(blockers)),
    }


def probe(
    *,
    repository: Path,
    config_path: Path,
    source_cards_path: Path,
    output_path: Path,
    fetcher: Callable[[str], dict[str, Any]] = _fetch,
    snapshot_overrides: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cards = json.loads(source_cards_path.read_text(encoding="utf-8"))["records"]
    snapshot_root = repository / SNAPSHOT_RELATIVE
    snapshot_root.mkdir(parents=True, exist_ok=True)
    page_cache: dict[str, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    snapshot_overrides = dict(snapshot_overrides or {})
    for card in sorted(cards, key=lambda item: str(item["source_uid"])):
        url = str(card["official_url"])
        if url not in page_cache:
            page_cache[url] = (
                _fetch_snapshot(url, snapshot_overrides[url])
                if url in snapshot_overrides
                else fetcher(url)
            )
        page = page_cache[url]
        page_without_payload = {key: value for key, value in page.items() if key != "payload"}
        if page.get("payload") and page.get("sha256"):
            snapshot_path = snapshot_root / f"{page['sha256']}.html"
            if not snapshot_path.exists():
                snapshot_path.write_bytes(page["payload"])
            metadata = _extract_html_metadata(page["payload"], str(page.get("final_url") or url))
            page_without_payload["snapshot_path"] = str(snapshot_path.relative_to(repository))
            page_without_payload["html_metadata"] = metadata
        results.append(_family_result(card, page_without_payload))

    track = config["round20_track_contracts"]["tracks"]["track_a_allen_full_landing"]
    report = {
        "schema_version": "neuronsifter-round20-allen-official-endpoint-probe-v1",
        "artifact_uid": "dt_round20_allen_official_endpoint_probe",
        "status": "metadata_probe_complete_with_rights_release_blockers",
        "retrieval_scope": "bounded_official_html_metadata_only_no_payload_retrieval",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "candidate_commit": _git(repository, "rev-parse", "HEAD"),
        "config_file_sha256": sha256_file(config_path),
        "source_cards_sha256": sha256_file(source_cards_path),
        "family_count": len(results),
        "expected_family_count": len(track["families"]),
        "families": results,
        "authorization": {
            "payload_download": False,
            "training": False,
            "evidence_fabric_sealing": False,
            "dt_core_evaluation": False,
        },
        "blocker_counts": {
            "dataset_specific_terms_unresolved": sum(
                "dataset_specific_terms_unresolved" in item["blockers"] for item in results
            ),
            "release_identity_unresolved": sum(
                "release_identity_unresolved" in item["blockers"] for item in results
            ),
            "endpoint_drift_or_error": sum(
                any(value in item["blockers"] for value in {
                    "official_endpoint_drift",
                    "official_endpoint_http_error",
                    "official_endpoint_probe_failed",
                })
                for item in results
            ),
        },
        "next_action": "resolve_current_official_landing_pages_and_dataset_specific_terms_then_freeze_exact_release_inventory",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--config", type=Path, default=CONFIG_RELATIVE)
    parser.add_argument("--source-cards", type=Path, default=SOURCE_CARDS_RELATIVE)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--offline-snapshot",
        action="append",
        default=[],
        metavar="URL=PATH",
        help="Use an already captured official HTML snapshot for URL; no network request is made.",
    )
    args = parser.parse_args()
    repository = args.repository_root.resolve()
    config_path = args.config if args.config.is_absolute() else repository / args.config
    source_cards_path = args.source_cards if args.source_cards.is_absolute() else repository / args.source_cards
    output_path = args.output or repository / "Output/audits/dt_round20_allen_official_endpoint_probe.json"
    output_path = output_path if output_path.is_absolute() else repository / output_path
    snapshot_overrides: dict[str, Path] = {}
    for item in args.offline_snapshot:
        if "=" not in item:
            raise SystemExit(f"invalid --offline-snapshot value: {item}")
        url, raw_path = item.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"offline snapshot is missing: {path}")
        snapshot_overrides[url] = path
    report = probe(
        repository=repository,
        config_path=config_path.resolve(),
        source_cards_path=source_cards_path.resolve(),
        output_path=output_path.resolve(),
        snapshot_overrides=snapshot_overrides,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "family_count": report["family_count"],
                "blocker_counts": report["blocker_counts"],
                "output": str(output_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
