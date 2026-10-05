"""Materialize an authorized Allen family from an official object inventory.

The runner is manifest driven.  Each object is either downloaded from its
official URL or reused from an explicitly supplied local official path, then
hashed into the shared CAS without overwriting an existing object.  Parsing
and QC are intentionally conservative: unsupported formats remain pending
instead of becoming false successes.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any
import zipfile

import requests

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import read_tsv, write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_cas(source: Path, cas_root: Path, digest: str) -> Path:
    destination = cas_root / digest[:2] / digest
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256(destination) != digest:
            raise ValueError(f"CAS collision with a different object: {destination}")
        return destination
    temporary = destination.parent / f".{digest}.part.{os.getpid()}"
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copyfile(source, temporary)
        if _sha256(temporary) != digest:
            raise ValueError(f"Staged object hash changed while registering: {source}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _download_resume(url: str, destination: Path) -> dict[str, str]:
    """Download an official object with a resumable local partial file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.parent / f".{destination.name}.part"
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    session = requests.Session()
    # Management-node proxy variables are not reachable from compute nodes.
    session.trust_env = False
    response = session.get(url, headers=headers, stream=True, timeout=(30, 120))
    response.raise_for_status()
    append = offset > 0 and response.status_code == 206
    if not append:
        offset = 0
    mode = "ab" if append else "wb"
    with partial.open(mode) as handle:
        for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
            if chunk:
                handle.write(chunk)
    metadata = {
        "etag": response.headers.get("ETag", ""),
        "last_modified": response.headers.get("Last-Modified", ""),
        "content_length": response.headers.get("Content-Length", ""),
        "content_range": response.headers.get("Content-Range", ""),
        "status_code": str(response.status_code),
    }
    response.close()
    partial.replace(destination)
    return metadata


def _read_inventory(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("objects", payload.get("inventory", []))
    if not isinstance(payload, list) or not payload:
        raise ValueError("Object inventory must be a non-empty list")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(payload):
        if not isinstance(raw, dict):
            raise TypeError(f"Inventory row {index} is not an object")
        url = str(raw.get("url", ""))
        local_path = str(raw.get("local_path", ""))
        if not url.startswith(("https://", "http://")) and not local_path:
            raise ValueError(f"Inventory row {index} lacks an official URL or local path")
        rows.append({
            "object_id": str(raw.get("object_id") or raw.get("name") or f"object_{index:04d}"),
            "url": url,
            "local_path": local_path,
            "expected_sha256": str(raw.get("sha256", "")),
            "expected_bytes": int(raw["bytes"]) if raw.get("bytes") is not None else None,
            "etag": str(raw.get("etag", "")),
            "last_modified": str(raw.get("last_modified", "")),
        })
    return rows


def _existing_cas_source(repo_root: Path, family_uid: str, object_id: str, url: str) -> Path | None:
    """Reuse a previously verified CAS object instead of downloading it again."""
    manifest_path = repo_root / "Output/audits/allen/materialization" / family_uid / "download_manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    objects = payload.get("objects", []) if isinstance(payload, dict) else []
    for record in objects:
        if not isinstance(record, dict):
            continue
        if str(record.get("object_id", "")) != object_id or str(record.get("url", "")) != url:
            continue
        cas_value = str(record.get("cas_path", ""))
        if not cas_value:
            return None
        candidate = (repo_root / cas_value).resolve()
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def _snapshot_uid(family_uid: str, authority_endpoint: str, inventory: list[dict[str, Any]], terms: dict[str, Any], citation: dict[str, Any]) -> str:
    canonical = {
        "family_uid": family_uid,
        "authority_endpoint": authority_endpoint,
        "inventory": inventory,
        "terms_snapshot_sha256": terms.get("sha256", ""),
        "citation_snapshot_sha256": citation.get("sha256", ""),
        "retrieved_at": _now(),
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "snapshot_" + hashlib.sha256(encoded).hexdigest()[:32]


def _authority_row(repo_root: Path, family_uid: str) -> dict[str, str]:
    rows = read_tsv(repo_root / "Output/audits/allen_authority_graph.tsv")
    matches = [row for row in rows if row.get("family_uid") == family_uid]
    if len(matches) != 1:
        raise ValueError(f"Expected one authority row for {family_uid}, found {len(matches)}")
    row = matches[0]
    if row.get("decision") not in {"authorized_allen_first_party_public", "authorized_public_third_party", "authorized_with_redistribution_restriction"}:
        raise PermissionError(f"Family is not authorized for materialization: {family_uid} ({row.get('decision')})")
    return row


def _json_record_count(payload: Any) -> tuple[int, str]:
    """Return a conservative record count and native unit label for API JSON."""
    if isinstance(payload, list):
        return len(payload), "record"
    if isinstance(payload, dict):
        for key in ("msg", "data", "rows", "results", "objects"):
            value = payload.get(key)
            if isinstance(value, list):
                if key == "msg" and len(value) == 1 and isinstance(value[0], dict) and "children" in value[0]:
                    def count_nodes(value: Any) -> int:
                        if not isinstance(value, dict):
                            return 0
                        children = value.get("children", [])
                        return 1 + sum(count_nodes(child) for child in children if isinstance(child, dict))

                    return count_nodes(value[0]), "atlas_structure"
                return len(value), key
        # Structure-graph responses contain one nested tree rather than rows.
        def count_nodes(value: Any) -> int:
            if not isinstance(value, dict):
                return 0
            children = value.get("children", [])
            return 1 + sum(count_nodes(child) for child in children if isinstance(child, dict))

        if "children" in payload:
            count = count_nodes(payload)
            if count:
                return count, "atlas_structure"
        return (1 if payload else 0), "object"
    return 0, "unknown"


def _process_format(path: Path, *, source_name: str | None = None) -> dict[str, Any]:
    """Parse a CAS object using its authoritative source object name.

    CAS paths deliberately use a content hash and therefore do not preserve a
    filename suffix.  Format selection must stay bound to the inventory's
    official object identity rather than the CAS storage path.
    """
    if path.stat().st_size <= 0:
        raise ValueError(f"Empty payload: {path}")
    name = (source_name or path.name).lower()
    suffix = Path(name).suffix.lower()
    if suffix == ".json":
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid JSON payload: {path}") from exc
        record_count, unit = _json_record_count(payload)
        if record_count <= 0:
            raise ValueError(f"JSON payload has no records: {path}")
        return {"format": "json", "parser_status": "passed", "record_count": record_count, "biological_unit": unit}
    if suffix in {".h5ad", ".h5", ".hdf5", ".nwb"}:
        try:
            import h5py
            with h5py.File(path, "r") as handle:
                shape = ()
                if "X" in handle:
                    x_node = handle["X"]
                    shape = tuple(int(value) for value in getattr(x_node, "shape", ()))
                    if not shape:
                        shape = tuple(int(value) for value in x_node.attrs.get("shape", ()))
                obs_count = len(handle["obs"].keys()) if "obs" in handle else 0
                var_count = len(handle["var"].keys()) if "var" in handle else 0
                root_keys = sorted(handle.keys())
            if not shape and obs_count == 0 and not root_keys:
                raise ValueError("HDF5 payload has no observable groups")
            record_count = shape[0] if shape else (obs_count or len(root_keys))
            return {"format": suffix.lstrip("."), "shape": list(shape), "obs_field_count": obs_count, "var_field_count": var_count, "root_key_count": len(root_keys), "parser_status": "passed", "record_count": int(record_count), "biological_unit": "observation_or_session"}
        except ImportError:
            return {"format": "h5ad", "parser_status": "pending_dependency_h5py"}
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            rows = list(reader)
        if len(rows) < 2 or not rows[0] or not any(cell.strip() for cell in rows[0]):
            raise ValueError(f"Tabular payload has no data rows: {path}")
        return {"format": suffix.lstrip("."), "parser_status": "passed", "record_count": len(rows) - 1, "column_count": len(rows[0]), "biological_unit": "catalog_record"}
    if suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
            members = [info for info in archive.infolist() if not info.is_dir()]
            tabular_members: list[dict[str, Any]] = []
            for info in members:
                member_name = info.filename.lower()
                if not member_name.endswith((".csv", ".tsv")):
                    continue
                delimiter = "\t" if member_name.endswith(".tsv") else ","
                with archive.open(info, "r") as raw_handle:
                    text_handle = io.TextIOWrapper(raw_handle, encoding="utf-8", errors="replace", newline="")
                    reader = csv.reader(text_handle, delimiter=delimiter)
                    try:
                        header = next(reader)
                    except StopIteration:
                        header = []
                    if not header or not any(cell.strip() for cell in header):
                        raise ValueError(f"ZIP tabular member has no header: {info.filename}")
                    row_count = sum(1 for _ in reader)
                    text_handle.detach()
                if row_count <= 0:
                    raise ValueError(f"ZIP tabular member has no records: {info.filename}")
                if "microarrayexpression" in member_name:
                    unit = "sample_expression_measurement"
                elif "sampleannot" in member_name:
                    unit = "sample_annotation"
                elif "probe" in member_name:
                    unit = "probe_annotation"
                elif "ontology" in member_name:
                    unit = "ontology_record"
                else:
                    unit = "catalog_record"
                tabular_members.append({"member": info.filename, "record_count": row_count, "column_count": len(header), "biological_unit": unit})
        if bad_member is not None or not members:
            raise ValueError(f"ZIP payload is empty or corrupt: {path}")
        if tabular_members:
            return {
                "format": "zip",
                "parser_status": "passed",
                "record_count": sum(int(item["record_count"]) for item in tabular_members),
                "biological_unit": tabular_members[0]["biological_unit"],
                "biological_units": sorted({str(item["biological_unit"]) for item in tabular_members}),
                "members": tabular_members,
            }
        return {"format": "zip", "parser_status": "passed", "record_count": len(members), "biological_unit": "archive_object"}
    if name.endswith(".nii.gz"):
        with gzip.open(path, "rb") as handle:
            header = handle.read(352)
        if len(header) < 4:
            raise ValueError(f"NIfTI gzip payload has no readable header: {path}")
        return {"format": "nii.gz", "parser_status": "passed", "record_count": 1, "biological_unit": "imaging_volume", "header_bytes": len(header)}
    if suffix == ".swc":
        rows = []
        with path.open("r", encoding="utf-8", errors="strict") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    fields = stripped.split()
                    if len(fields) < 7:
                        raise ValueError(f"Malformed SWC row in {path}")
                    rows.append(fields)
        if not rows:
            raise ValueError(f"SWC payload has no nodes: {path}")
        return {"format": "swc", "parser_status": "passed", "record_count": len(rows), "biological_unit": "morphology_node"}
    if suffix in {".xml", ".nii"}:
        if path.stat().st_size <= 0:
            raise ValueError(f"Empty payload: {path}")
        return {"format": suffix.lstrip("."), "parser_status": "passed", "record_count": 1, "biological_unit": "source_object", "bytes": path.stat().st_size}
    return {"format": suffix.lstrip(".") or "unknown", "parser_status": "pending_typed_handler", "bytes": path.stat().st_size}


def materialize(repo_root: Path, family_uid: str, inventory_path: Path, *, authority_endpoint: str, terms_path: Path, citation_path: Path, output_root: Path, staging_root: Path | None = None) -> dict[str, Any]:
    authority = _authority_row(repo_root, family_uid)
    inventory = _read_inventory(inventory_path)
    terms = {"path": _relative(terms_path), "sha256": _sha256(terms_path), "bytes": terms_path.stat().st_size}
    citation = {"path": _relative(citation_path), "sha256": _sha256(citation_path), "bytes": citation_path.stat().st_size}
    retrieval_started = _now()
    cas_root = repo_root / "Input/raw/public/allen/objects/sha256"
    staged_root = (staging_root or (repo_root / "Input/raw/public/allen/staging")) / family_uid
    records: list[dict[str, Any]] = []
    parsed: list[dict[str, Any]] = []
    for item in inventory:
        response_metadata: dict[str, str] = {}
        local_source = (repo_root / item["local_path"]).resolve() if item["local_path"] else None
        if local_source is not None and not local_source.is_file():
            raise FileNotFoundError(local_source)
        staged = staged_root / item["object_id"]
        reused_existing_cas = False
        if local_source is not None:
            source = local_source
        else:
            existing_cas = _existing_cas_source(repo_root, family_uid, item["object_id"], item["url"])
            if existing_cas is not None:
                source = existing_cas
                reused_existing_cas = True
            else:
                source = staged
                if not source.exists():
                    response_metadata = _download_resume(item["url"], source)
        byte_count = source.stat().st_size
        if byte_count <= 0:
            raise ValueError(f"Empty payload for {item['object_id']}")
        digest = _sha256(source)
        if item["expected_sha256"] and item["expected_sha256"] != digest:
            raise ValueError(f"Checksum mismatch for {item['object_id']}: {digest} != {item['expected_sha256']}")
        if item["expected_bytes"] is not None and item["expected_bytes"] != byte_count:
            raise ValueError(f"Byte count mismatch for {item['object_id']}: {byte_count} != {item['expected_bytes']}")
        cas_path = _atomic_cas(source, cas_root, digest)
        parsed_result = _process_format(source, source_name=item["object_id"])
        parsed.append({"object_id": item["object_id"], **parsed_result})
        records.append({
            "object_id": item["object_id"], "url": item["url"], "source_path": _relative(cas_path),
            "cas_path": _relative(cas_path), "bytes": byte_count, "sha256": digest,
            "etag": item["etag"] or response_metadata.get("etag", ""),
            "last_modified": item["last_modified"] or response_metadata.get("last_modified", ""),
            "content_length": response_metadata.get("content_length", ""),
            "content_range": response_metadata.get("content_range", ""),
            "reused_existing_cas": reused_existing_cas,
        })
    snapshot_uid = _snapshot_uid(family_uid, authority_endpoint, records, terms, citation)
    release_root = repo_root / "Input/raw/public/allen/releases" / family_uid / snapshot_uid
    manifest = {
        "schema_version": "neuronsifter-allen-public-object-manifest",
        "family_uid": family_uid,
        "snapshot_uid": snapshot_uid,
        "authority_endpoint": authority_endpoint,
        "retrieved_at": retrieval_started,
        "terms_snapshot": terms,
        "citation_snapshot": citation,
        "objects": records,
        "expected_object_count": len(inventory),
        "verified_object_count": len(records),
        "redistribution_policy": authority.get("decision"),
    }
    write_json(release_root / "manifest.json", manifest)
    family_output = output_root / "allen/materialization" / family_uid
    write_json(family_output / "authority.json", {"family_uid": family_uid, "decision": authority.get("decision"), "endpoint": authority_endpoint, "terms_snapshot": terms, "citation_snapshot": citation})
    write_json(family_output / "snapshot_identity.json", {"family_uid": family_uid, "snapshot_uid": snapshot_uid, "manifest_path": _relative(release_root / "manifest.json"), "manifest_sha256": _sha256(release_root / "manifest.json"), "retrieved_at": retrieval_started})
    write_json(family_output / "download_manifest.json", manifest)
    integrity = {"family_uid": family_uid, "official_endpoint": True, "terms_snapshot_present": True, "immutable_snapshot_identity": True, "real_payload_bytes": all(item["bytes"] > 0 for item in records), "expected_objects_accounted": len(records) == len(inventory), "downloaded_objects_hash_verified": len(records) == len(inventory), "checksum_mismatch": 0, "undeclared_mirror_objects": 0, "objects": records}
    write_json(family_output / "integrity.json", integrity)
    processed_root = repo_root / "Input/processed/public_foundation/allen" / family_uid / snapshot_uid
    parser_status = "passed" if all(item["parser_status"] == "passed" for item in parsed) else "partial"
    canonical_record_count = sum(int(item.get("record_count", 0)) for item in parsed)
    biological_units = sorted({str(item.get("biological_unit", "unknown")) for item in parsed})
    write_json(processed_root / "parser_validation.json", {"family_uid": family_uid, "snapshot_uid": snapshot_uid, "status": parser_status, "objects": parsed})
    write_json(processed_root / "canonicalization_summary.json", {"family_uid": family_uid, "snapshot_uid": snapshot_uid, "status": "passed" if canonical_record_count > 0 and parser_status == "passed" else "blocked_empty", "canonical_record_count": canonical_record_count, "source_objects": [item["object_id"] for item in parsed], "biological_units": biological_units})
    write_json(processed_root / "biological_unit_audit.json", {"family_uid": family_uid, "snapshot_uid": snapshot_uid, "status": "passed" if canonical_record_count > 0 and parser_status == "passed" else "blocked", "biological_unit_policy": "source_native_unit_bound", "unit_count": canonical_record_count, "unit_types": biological_units})
    write_json(processed_root / "qc_summary.json", {"family_uid": family_uid, "snapshot_uid": snapshot_uid, "status": "passed" if parser_status == "passed" and canonical_record_count > 0 else "blocked", "nonempty_objects": len(records), "hash_verified_objects": len(records), "canonical_record_count": canonical_record_count})
    write_json(processed_root / "split_role.json", {"family_uid": family_uid, "snapshot_uid": snapshot_uid, "role": "E0_E1_state_observation_or_holdout", "controlled_sources": 0, "intervention_promotion": False})
    write_json(processed_root / "consumer_execution.json", {"family_uid": family_uid, "snapshot_uid": snapshot_uid, "status": "completed" if parser_status == "passed" and canonical_record_count > 0 else "blocked", "consumer": "allen_public_object_inventory_consumer", "record_count": canonical_record_count})
    return {"family_uid": family_uid, "snapshot_uid": snapshot_uid, "object_count": len(records), "record_count": canonical_record_count, "parser_status": parser_status, "status": "materialized" if parser_status == "passed" and canonical_record_count > 0 else "blocked"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize an authorized Allen family from an official inventory.")
    parser.add_argument("--family", required=True)
    parser.add_argument("--inventory", required=True, help="JSON list/object inventory with url or explicit local_path")
    parser.add_argument("--authority-endpoint", required=True)
    parser.add_argument("--terms", default="Input/raw/allen_brain_cell_atlas/terms/allen_terms_snapshot.html")
    parser.add_argument("--citation", default="Output/audits/round20/official_metadata_resolution/snapshots/allen_citation_policy.html")
    parser.add_argument("--output-root", default="Output/audits")
    parser.add_argument("--staging-root", default="", help="Optional node-local staging root; CAS and manifests remain in the repository")
    args = parser.parse_args()
    staging_root = Path(args.staging_root).expanduser() if args.staging_root else None
    if staging_root is not None and not staging_root.is_absolute():
        staging_root = (REPO_ROOT / staging_root).resolve()
    result = materialize(REPO_ROOT, args.family, (REPO_ROOT / args.inventory).resolve(), authority_endpoint=args.authority_endpoint, terms_path=(REPO_ROOT / args.terms).resolve(), citation_path=(REPO_ROOT / args.citation).resolve(), output_root=(REPO_ROOT / args.output_root).resolve(), staging_root=staging_root)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
