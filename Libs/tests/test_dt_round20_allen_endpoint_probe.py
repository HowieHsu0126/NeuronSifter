"""Fail-closed tests for the bounded Allen official endpoint probe."""

from __future__ import annotations

from pathlib import Path

import yaml

from Libs.scripts.validators import probe_dt_round20_allen_official_endpoints as probe


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_probe_rejects_non_official_hosts_and_extracts_only_official_links() -> None:
    assert probe._host_allowed("alleninstitute.org") is True
    assert probe._host_allowed("sub.brain-map.org") is True
    assert probe._host_allowed("example.org") is False
    metadata = probe._extract_html_metadata(
        b'<title>Atlas release 2025</title>'
        b'<a href="https://atlas.brain-map.org/atlas?version=2025">atlas</a>'
        b'<a href="https://example.org/payload">payload</a>',
        "https://atlas.brain-map.org/",
    )
    assert metadata["title"] == "Atlas release 2025"
    assert metadata["same_authority_links"] == [
        "https://atlas.brain-map.org/atlas?version=2025"
    ]
    assert metadata["release_identity_promoted"] is False
    assert metadata["release_candidate_tokens"]


def test_probe_preserves_historical_listing_evidence_without_promoting_uid() -> None:
    card = {
        "source_uid": "allen_abc_human",
        "official_url": "https://alleninstitute.org/current",
        "release_listing_receipts": [
            {
                "relative_path": "Input/raw/listing.xml",
                "sha256": "a" * 64,
                "release_hint": "WHB-10Xv3/20241115",
                "listed_object_count": 1,
            }
        ],
    }
    page = {
        "status": "official_endpoint_drift",
        "http_status": 404,
        "final_url": "https://alleninstitute.org/current",
        "content_type": "text/html",
        "size_bytes": 0,
        "sha256": None,
        "etag": None,
        "last_modified": None,
        "error": None,
    }
    result = probe._family_result(card, page)
    assert result["release_status"] == "historical_listing_evidence_present_exact_uid_not_promoted"
    assert result["release_identity_promoted"] is False
    assert "dataset_specific_terms_unresolved" in result["blockers"]
    assert "official_endpoint_drift" in result["blockers"]


def test_offline_snapshot_is_bounded_and_hash_bound(tmp_path: Path) -> None:
    snapshot = tmp_path / "page.html"
    snapshot.write_bytes(b"<title>official</title>")
    result = probe._fetch_snapshot("https://atlas.brain-map.org/", snapshot)
    assert result["retrieval_mode"] == "offline_snapshot"
    assert result["sha256"] == probe.sha256_file(snapshot)
    assert result["payload"] == b"<title>official</title>"


def test_probe_is_read_only_with_respect_to_payload_authorization() -> None:
    assert yaml.safe_load(
        (REPO_ROOT / "Libs/configs/validation/round20_track_contracts.yaml").read_text()
    )["round20_track_contracts"]["tracks"]["track_a_allen_full_landing"]["authorization"][
        "payload_download"
    ] is False
