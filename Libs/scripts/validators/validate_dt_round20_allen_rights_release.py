#!/usr/bin/env python3
"""Freeze bounded official Allen terms metadata and fail-closed family decisions."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping
from urllib.request import Request, urlopen

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_RELATIVE = Path("Libs/configs/validation/round20_track_contracts.yaml")
TERMS_URL = "https://alleninstitute.org/terms-of-use/"
CITATION_URL = "https://alleninstitute.org/legal/citation-policy"
MAX_SNAPSHOT_BYTES = 2 * 1024 * 1024


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _git(repository: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repository, check=True, capture_output=True, text=True
    ).stdout.strip()


def _fetch(url: str) -> tuple[bytes, dict[str, str], int | None, str | None]:
    request = Request(url, headers={"User-Agent": "NeuronSifter-Round20-metadata-audit/1.0"})
    try:
        with urlopen(request, timeout=25) as response:
            payload = response.read(MAX_SNAPSHOT_BYTES + 1)
            headers = {str(key).lower(): str(value) for key, value in response.headers.items()}
            if len(payload) > MAX_SNAPSHOT_BYTES:
                return b"", headers, int(response.status), "snapshot_exceeds_bound"
            return payload, headers, int(response.status), None
    except Exception as exc:  # network state is part of the auditable result
        return b"", {}, None, f"fetch_error:{type(exc).__name__}:{exc}"


def _snapshot(repository: Path, snapshot_root: Path, name: str, url: str) -> dict[str, Any]:
    payload, headers, status, error = _fetch(url)
    if payload:
        path = snapshot_root / f"{name}.html"
        path.write_bytes(payload)
        text = payload.decode("utf-8", errors="replace").lower()
        markers = {
            "noncommercial_language": "noncommercial" in text,
            "citation_language": "citation policy" in text,
            "redistribution_language": "redistribut" in text,
        }
        return {
            "url": url,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "http_status": status,
            "path": str(path.relative_to(repository)),
            "size_bytes": len(payload),
            "sha256": sha256_bytes(payload),
            "etag": headers.get("etag"),
            "last_modified": headers.get("last-modified"),
            "markers": markers,
            "error": error,
        }
    return {
        "url": url,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "http_status": status,
        "path": None,
        "size_bytes": 0,
        "sha256": None,
        "etag": headers.get("etag"),
        "last_modified": headers.get("last-modified"),
        "markers": {},
        "error": error or "empty_snapshot",
    }


def _snapshot_from_file(
    repository: Path, snapshot_root: Path, name: str, url: str, source: Path
) -> dict[str, Any]:
    snapshot_root.mkdir(parents=True, exist_ok=True)
    payload = source.read_bytes()
    if len(payload) > MAX_SNAPSHOT_BYTES:
        return {
            "url": url,
            "retrieval_method": "local_snapshot",
            "path": None,
            "size_bytes": len(payload),
            "sha256": None,
            "markers": {},
            "error": "snapshot_exceeds_bound",
        }
    target = snapshot_root / f"{name}.html"
    if source.resolve() != target.resolve():
        target.write_bytes(payload)
    text = payload.decode("utf-8", errors="replace").lower()
    return {
        "url": url,
        "retrieval_method": "local_snapshot",
        "source_path": str(source.relative_to(repository)) if source.is_relative_to(repository) else str(source),
        "path": str(target.relative_to(repository)),
        "size_bytes": len(payload),
        "sha256": sha256_bytes(payload),
        "markers": {
            "noncommercial_language": "noncommercial" in text,
            "citation_language": "citation policy" in text,
            "redistribution_language": "redistribut" in text,
        },
        "error": None,
    }


def audit(
    *, repository: Path, config_path: Path, terms_snapshot: Path | None, citation_snapshot: Path | None
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    track = config["round20_track_contracts"]["tracks"]["track_a_allen_full_landing"]
    source_cards_path = repository / "Output/audits/round20/historical_inputs/allen_source_cards.json"
    cards = json.loads(source_cards_path.read_text(encoding="utf-8"))["records"]
    snapshot_root = repository / "Output/audits/round20/allen_terms_snapshots"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    terms = (
        _snapshot_from_file(repository, snapshot_root, "allen_terms_of_use", TERMS_URL, terms_snapshot)
        if terms_snapshot is not None
        else _snapshot(repository, snapshot_root, "allen_terms_of_use", TERMS_URL)
    )
    citation = (
        _snapshot_from_file(repository, snapshot_root, "allen_citation_policy", CITATION_URL, citation_snapshot)
        if citation_snapshot is not None
        else _snapshot(repository, snapshot_root, "allen_citation_policy", CITATION_URL)
    )
    snapshots = {"terms_of_use": terms, "citation_policy": citation}
    blockers: list[dict[str, Any]] = []
    for name, snapshot in snapshots.items():
        if snapshot["sha256"] is None:
            blockers.append({"code": "terms_snapshot_unavailable", "snapshot": name, "error": snapshot["error"]})
    families: list[dict[str, Any]] = []
    for card in sorted(cards, key=lambda item: str(item["source_uid"])):
        source_uid = str(card["source_uid"])
        listing_receipts = list(card.get("release_listing_receipts") or [])
        release_identity = card.get("source_release_uid") or card.get("release_version")
        if not listing_receipts and release_identity in {None, "", "release_identity_pending"}:
            decision = "blocked_missing_release_identity"
            blocker_code = "blocked_missing_release_identity"
        else:
            decision = "blocked_terms_ambiguity"
            blocker_code = "blocked_terms_ambiguity"
        families.append(
            {
                "family_uid": source_uid,
                "dataset_and_release": {
                    "release_uid": card.get("source_release_uid"),
                    "release_version": card.get("release_version"),
                    "listing_receipt_count": len(listing_receipts),
                },
                "official_host": card.get("official_url"),
                "payload_host": card.get("official_url"),
                "governing_terms": {"url": TERMS_URL, "snapshot_sha256": terms["sha256"]},
                "dataset_specific_license": "unresolved",
                "third_party_license": "unresolved",
                "citation_policy": {"url": CITATION_URL, "snapshot_sha256": citation["sha256"]},
                "internal_download_allowed": "not_resolved",
                "internal_transformation_allowed": "not_resolved",
                "internal_training_allowed": "not_resolved",
                "aggregate_reporting_allowed": "not_resolved",
                "raw_redistribution_allowed": "not_resolved",
                "processed_redistribution_allowed": "not_resolved",
                "derived_feature_release_allowed": "not_resolved",
                "checkpoint_release_allowed": "not_resolved",
                "commercial_use_status": "noncommercial_general_terms_only",
                "required_attribution": "Allen Institute citation policy; dataset-specific citation pending",
                "decision": decision,
                "blocker": blocker_code,
                "terms_snapshot_sha256": terms["sha256"],
                "content_hash": sha256_bytes(json.dumps(card, sort_keys=True).encode()),
            }
        )

    manifest_root = repository / "Input/manifests/allen"
    manifest_root.mkdir(parents=True, exist_ok=True)
    inventory_path = manifest_root / "allen_release_inventory.tsv"
    inventory_fields = [
        "release_uid",
        "release_date_or_version",
        "official_landing_page",
        "api_or_download_endpoint",
        "object_inventory",
        "format",
        "expected_size",
        "official_checksum",
        "expected_biological_object_counts",
        "pagination_or_request_identity",
        "etag_or_last_modified",
        "family_uid",
        "status",
    ]
    with inventory_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=inventory_fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for card, family in zip(sorted(cards, key=lambda item: str(item["source_uid"])), families):
            receipts = list(card.get("release_listing_receipts") or [])
            hints = sorted(str(item.get("release_hint")) for item in receipts if item.get("release_hint"))
            writer.writerow(
                {
                    "release_uid": "",
                    "release_date_or_version": "|".join(hints) or "release_identity_pending",
                    "official_landing_page": card.get("official_url", ""),
                    "api_or_download_endpoint": card.get("official_url", ""),
                    "object_inventory": json.dumps(receipts, sort_keys=True, separators=(",", ":")),
                    "format": "listing_xml_only" if receipts else "not_discovered",
                    "expected_size": "unknown",
                    "official_checksum": "unknown",
                    "expected_biological_object_counts": "unknown",
                    "pagination_or_request_identity": "|".join(str(item.get("relative_path")) for item in receipts) or "pending",
                    "etag_or_last_modified": "unknown",
                    "family_uid": family["family_uid"],
                    "status": family["decision"],
                }
            )
    retrieval_manifest_path = manifest_root / "allen_retrieval_manifest.json"
    retrieval_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "neuronsifter-round20-allen-retrieval-manifest-v1",
                "status": "blocked_before_payload_retrieval",
                "payload_retrieval_authorized": False,
                "source_release_inventory": str(inventory_path.relative_to(repository)),
                "families": [
                    {
                        "family_uid": row["family_uid"],
                        "retrieve": False,
                        "reason": row["blocker"],
                        "release_uid": row["dataset_and_release"]["release_uid"],
                    }
                    for row in families
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    if len(families) != len(track["families"]):
        blockers.append({"code": "family_count_mismatch", "expected": len(track["families"]), "observed": len(families)})
    if not terms["markers"].get("noncommercial_language") or not terms["markers"].get("citation_language"):
        blockers.append({"code": "general_terms_markers_missing"})
    return {
        "schema_version": "neuronsifter-round20-allen-rights-release-v1",
        "artifact_uid": "dt_round20_allen_rights_matrix",
        "status": "blocked_current_rights_or_release_resolution",
        "candidate_commit": _git(repository, "rev-parse", "HEAD"),
        "config_file_sha256": sha256_file(config_path),
        "retrieval_scope": "bounded_official_metadata_only_no_payload_retrieval",
        "snapshots": snapshots,
        "families": families,
        "release_inventory": {
            "path": str(inventory_path.relative_to(repository)),
            "sha256": sha256_file(inventory_path),
            "family_count": len(families),
            "frozen_release_count": sum(bool(row["dataset_and_release"]["release_uid"]) for row in families),
        },
        "retrieval_manifest": {
            "path": str(retrieval_manifest_path.relative_to(repository)),
            "sha256": sha256_file(retrieval_manifest_path),
            "payload_retrieval_authorized": False,
        },
        "decision_counts": {
            decision: sum(row["decision"] == decision for row in families)
            for decision in (
                "approved_internal_noncommercial",
                "approved_with_release_restrictions",
                "approved_internal_and_derived_release",
                "blocked_third_party_terms",
                "blocked_missing_release_identity",
                "blocked_terms_ambiguity",
            )
        },
        "blockers": blockers,
        "authorization": {"payload_download": False, "training": False},
        "next_action": "resolve_dataset_specific_terms_and_freeze_exact_release_inventory_before_payload_retrieval",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--config", type=Path, default=CONFIG_RELATIVE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--terms-snapshot", type=Path)
    parser.add_argument("--citation-snapshot", type=Path)
    args = parser.parse_args()
    repository = args.repository_root.resolve()
    config_path = args.config if args.config.is_absolute() else repository / args.config
    output = args.output or repository / "Output/audits/dt_round20_allen_rights_matrix.json"
    output = output if output.is_absolute() else repository / output
    terms_snapshot = args.terms_snapshot.resolve() if args.terms_snapshot else None
    citation_snapshot = args.citation_snapshot.resolve() if args.citation_snapshot else None
    if terms_snapshot is not None and not terms_snapshot.is_file():
        raise SystemExit(f"terms snapshot is missing: {terms_snapshot}")
    if citation_snapshot is not None and not citation_snapshot.is_file():
        raise SystemExit(f"citation snapshot is missing: {citation_snapshot}")
    report = audit(
        repository=repository,
        config_path=config_path.resolve(),
        terms_snapshot=terms_snapshot,
        citation_snapshot=citation_snapshot,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "blockers": len(report["blockers"]), "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
