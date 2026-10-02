"""Resolve executable Allen public-materialization authority.

The builder binds the existing official endpoint, terms, and citation snapshots
to family-level decisions.  A conventional release tag is optional: an
official snapshot plus its source hash is a reproducible release identity.  No
email or permission-request state is consulted by the active route.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config, write_json, write_tsv


GRAPH_FIELDS = [
    "family_uid", "display_name", "official_product_record", "official_dataset_record",
    "release_locator", "payload_host", "release_identity_decision", "dataset_terms_decision",
    "third_party_terms_decision", "citation_record", "internal_materialization_authorized",
    "decision", "blockers",
]
RELEASE_FIELDS = [
    "family_uid", "display_name", "release_identity_decision", "release_identity",
    "official_snapshot_uid", "official_snapshot_sha256", "release_locator", "blockers",
]
TERMS_FIELDS = [
    "family_uid", "display_name", "dataset_terms_decision", "dataset_specific_license",
    "governing_terms", "internal_materialization_authorized", "blockers",
]
THIRD_PARTY_FIELDS = [
    "family_uid", "display_name", "payload_host", "third_party_terms_decision",
    "third_party_license", "internal_materialization_authorized", "blockers",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fail-closed Allen authority artifacts without downloading payloads.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--historical-matrix", default="Output/audits/dt_round20_allen_rights_matrix_v2.json")
    parser.add_argument("--output-root", default="Output/audits")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected an object in {path}")
    return payload


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe_binding(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "Output/audits/allen_authority_source_probe.json"
    if not path.is_file():
        return {"exists": False, "path": _relative(path), "received_source_count": 0, "failed_source_count": 0}
    payload = _read_json(path)
    sources = [
        source
        for record in payload.get("records", [])
        for source in dict(record.get("sources", {})).values()
    ]
    received = sum(source.get("transport_status") == "received" for source in sources)
    return {
        "exists": True,
        "path": _relative(path),
        "sha256": _sha256(path),
        "received_source_count": received,
        "failed_source_count": len(sources) - received,
        "rights_or_release_decision_changed": payload.get("rights_or_release_decision_changed"),
    }


def _is_authorized(record: Mapping[str, Any]) -> bool:
    return all(
        bool(record.get(field))
        for field in ("internal_download_allowed", "internal_transformation_allowed", "internal_training_allowed")
    )


def _release_decision(record: Mapping[str, Any], *, snapshot_available: bool) -> str:
    release_text = str(record.get("dataset_and_release", ""))
    if "CONFLICT" in release_text.upper():
        return "blocked_explicit_terms_conflict"
    if snapshot_available and record.get("official_snapshot_uid") and record.get("official_snapshot_sha256"):
        if "UNRESOLVED" in release_text or "NOT_FROZEN" in release_text:
            return "resolved_api_snapshot"
        return "resolved_immutable_release"
    return "blocked_identity_unreproducible"


def _terms_decision(record: Mapping[str, Any], *, terms_available: bool) -> str:
    text = str(record.get("dataset_specific_license", ""))
    if "CONFLICT" in text.upper() or "PROHIBIT" in text.upper():
        return "blocked_explicit_terms_conflict"
    if not terms_available:
        return "blocked_dataset_terms"
    return "resolved_internal_noncommercial_only"


def _third_party_decision(record: Mapping[str, Any], *, first_party_payload: bool) -> str:
    text = str(record.get("third_party_license", ""))
    if first_party_payload:
        return "not_applicable"
    if "CONFLICT" in text.upper() or "PROHIBIT" in text.upper():
        return "blocked_explicit_terms_conflict"
    if text and "UNRESOLVED" not in text.upper():
        return "authorized_public_third_party"
    return "blocked_third_party_terms_unavailable"


def _snapshot_bindings(repo_root: Path) -> dict[str, dict[str, str]]:
    candidates = {
        "terms": [
            repo_root / "Input/raw/allen_brain_cell_atlas/terms/allen_terms_snapshot.html",
            repo_root / "Output/audits/round20/official_metadata_resolution/snapshots/allen_terms_of_use.html",
        ],
        "citation": [repo_root / "Output/audits/round20/official_metadata_resolution/snapshots/allen_citation_policy.html"],
    }
    bindings: dict[str, dict[str, str]] = {}
    for name, paths in candidates.items():
        for path in paths:
            if path.is_file():
                bindings[name] = {"path": _relative(path), "sha256": _sha256(path), "bytes": str(path.stat().st_size)}
                break
    return bindings


def _first_party_payload(record: Mapping[str, Any], entry: Mapping[str, Any]) -> bool:
    """Recognize Allen-controlled endpoints, including the ABC S3 manifest."""
    endpoint_text = " ".join(str(record.get(key, "")) for key in ("payload_host", "official_host"))
    endpoint_text += " " + " ".join(str(entry.get(key, "")) for key in ("dataset_record", "release_locator"))
    lowered = endpoint_text.lower()
    return any(token in lowered for token in ("allen-brain-cell-atlas.s3", "alleninstitute", "brain-map.org", "brain-map.org"))


def build(repo_root: Path, historical_matrix: Path, output_root: Path) -> dict[str, Any]:
    config = load_config(repo_root / "Libs/configs/config_stack.yaml")
    catalog = config["allen_authority_catalog"]
    families = dict(catalog["families"])
    matrix = _read_json(historical_matrix)
    records = list(matrix.get("records", []))
    if len(records) != int(catalog["family_count"]):
        raise ValueError("Historical authority matrix does not contain the configured family count.")
    if set(families) != {str(record.get("family_uid")) for record in records}:
        raise ValueError("Authority catalog and historical matrix family identifiers differ.")

    graph_rows: list[dict[str, Any]] = []
    release_rows: list[dict[str, Any]] = []
    terms_rows: list[dict[str, Any]] = []
    third_party_rows: list[dict[str, Any]] = []
    snapshots = _snapshot_bindings(repo_root)
    terms_available = "terms" in snapshots
    citation_available = "citation" in snapshots
    for record in sorted(records, key=lambda item: str(item["family_uid"])):
        family_uid = str(record["family_uid"])
        entry = families[family_uid]
        first_party = _first_party_payload(record, entry)
        release = _release_decision(record, snapshot_available=terms_available)
        terms = _terms_decision(record, terms_available=terms_available)
        third_party = _third_party_decision(record, first_party_payload=first_party)
        explicit_blocker = str(record.get("blocker") or "")
        blockers = ";".join(filter(None, [
            "" if release.startswith("resolved") else release,
            "" if terms.startswith("resolved") else terms,
            "" if third_party == "not_applicable" or third_party.startswith("authorized") else third_party,
            "citation_snapshot_missing" if not citation_available else "",
        ]))
        authorized = release.startswith("resolved") and terms.startswith("resolved") and third_party.startswith("authorized") or (
            release.startswith("resolved") and terms.startswith("resolved") and third_party == "not_applicable"
        )
        decision = (
            "authorized_allen_first_party_public" if authorized and first_party else
            "authorized_public_third_party" if authorized else
            third_party if third_party != "not_applicable" else
            terms if terms.startswith("blocked") else release
        )
        graph_rows.append({
            "family_uid": family_uid,
            "display_name": entry["display_name"],
            "official_product_record": entry["product_record"],
            "official_dataset_record": entry["dataset_record"],
            "release_locator": entry["release_locator"],
            "payload_host": record.get("payload_host", ""),
            "release_identity_decision": release,
            "dataset_terms_decision": terms,
            "third_party_terms_decision": third_party,
            "citation_record": entry["citation_record"],
            "internal_materialization_authorized": authorized,
            "decision": decision,
            "blockers": blockers or explicit_blocker if not authorized else "",
        })
        release_rows.append({
            "family_uid": family_uid,
            "display_name": entry["display_name"],
            "release_identity_decision": release,
            "release_identity": record.get("dataset_and_release", "") if "UNRESOLVED" not in str(record.get("dataset_and_release", "")) else f"official_snapshot:{record.get('official_snapshot_uid', '')}",
            "official_snapshot_uid": record.get("official_snapshot_uid", ""),
            "official_snapshot_sha256": record.get("official_snapshot_sha256", ""),
            "release_locator": entry["release_locator"],
            "blockers": blockers,
        })
        terms_rows.append({
            "family_uid": family_uid,
            "display_name": entry["display_name"],
            "dataset_terms_decision": terms,
            "dataset_specific_license": record.get("dataset_specific_license", "") if "UNRESOLVED" not in str(record.get("dataset_specific_license", "")) else "allen_terms_of_use_snapshot",
            "governing_terms": record.get("governing_terms", "") or "allen_terms_of_use",
            "internal_materialization_authorized": authorized,
            "blockers": blockers or explicit_blocker if not authorized else "",
        })
        if third_party != "not_applicable" or family_uid.startswith("allen_abc_"):
            third_party_rows.append({
                "family_uid": family_uid,
                "display_name": entry["display_name"],
                "payload_host": record.get("payload_host", ""),
                "third_party_terms_decision": third_party,
                "third_party_license": (
                    record.get("third_party_license", "")
                    if third_party != "not_applicable"
                    else "not_applicable_official_allen_controlled_endpoint"
                ),
                "internal_materialization_authorized": authorized,
                "blockers": blockers or explicit_blocker if not authorized else "",
            })

    output_root.mkdir(parents=True, exist_ok=True)
    write_tsv(output_root / "allen_authority_graph.tsv", GRAPH_FIELDS, graph_rows)
    write_tsv(output_root / "allen_release_identity_resolution.tsv", RELEASE_FIELDS, release_rows)
    write_tsv(output_root / "allen_dataset_terms_resolution.tsv", TERMS_FIELDS, terms_rows)
    write_tsv(output_root / "allen_third_party_payload_terms.tsv", THIRD_PARTY_FIELDS, third_party_rows)
    authorized_families = [row["family_uid"] for row in graph_rows if row["internal_materialization_authorized"]]
    probe_binding = _probe_binding(repo_root)
    inventory_contract = {
        "schema_version": "neuronsifter-allen-immutable-inventory-contract",
        "status": "ready_for_progressive_materialization" if authorized_families else "blocked",
        "historical_matrix": _relative(historical_matrix),
        "authority_graph": _relative(output_root / "allen_authority_graph.tsv"),
        "official_source_probe": probe_binding,
        "authorized_families": authorized_families,
        "download_permission_required": False,
        "atomic_cas_required": True,
        "checksum_or_response_hash_required": True,
        "empty_payload_is_completion": False,
        "public_only": True,
        "e2_or_e4_promotion_allowed": False,
        "terms_snapshot": snapshots.get("terms", {}),
        "citation_snapshot": snapshots.get("citation", {}),
        "authorization_semantics": "official_public_endpoint_with_terms_snapshot_supports_internal_noncommercial_materialization",
    }
    gate = {
        "schema_version": "neuronsifter-allen-rights-release-gate",
        "status": "blocked" if not authorized_families else "progressive_authority_resolution",
        "families": len(graph_rows),
        "release_identity_decisions": len(release_rows),
        "dataset_terms_decisions": len(terms_rows),
        "third_party_terms_decisions": len(third_party_rows),
        "families_authorized_for_internal_materialization": len(authorized_families),
        "unknown_decisions": 0,
        "payload_download_authorized": bool(authorized_families),
        "training_authorized": False,
        "evidence_fabric_sealed": False,
        "dt_core_evaluated": False,
        "next_action": "materialize_authorized_families_from_official_endpoints",
        "official_source_probe": probe_binding,
        "terms_snapshot": snapshots.get("terms", {}),
        "citation_snapshot": snapshots.get("citation", {}),
    }
    write_json(output_root / "allen_immutable_inventory_contract.json", inventory_contract)
    write_json(output_root / "allen_rights_release_gate.json", gate)
    return {"graph_rows": graph_rows, "gate": gate, "inventory_contract": inventory_contract}


def main() -> None:
    args = _parse_args()
    result = build(
        REPO_ROOT,
        (REPO_ROOT / args.historical_matrix).resolve(),
        (REPO_ROOT / args.output_root).resolve(),
    )
    print(json.dumps({"status": result["gate"]["status"], "families": result["gate"]["families"]}, sort_keys=True))


if __name__ == "__main__":
    main()
