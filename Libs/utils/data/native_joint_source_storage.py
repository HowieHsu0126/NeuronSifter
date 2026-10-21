"""Lossless gzip JSON for one complete native joint group; no cross-call cache."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from Libs.utils.data.raw_data_registry import resolve_repo_path
from Libs.utils.data.native_input_staging import NativeInputLocations


def write_joint_source_reference(path: Path, source: dict, *, max_bytes: int) -> dict:
    payload = json.dumps(source, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    compressed = gzip.compress(payload, compresslevel=6, mtime=0)
    if max_bytes <= 0 or len(compressed) > max_bytes:
        raise ValueError("Complete joint source exceeds its declared compressed byte ceiling")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(compressed)
    return {"source_posterior_uid": source["source_posterior_uid"], "path": str(path.resolve()),
            "compression": "gzip", "sha256": hashlib.sha256(compressed).hexdigest(),
            "size_bytes": len(compressed), "uncompressed_size_bytes": len(payload)}


def read_joint_source_reference(reference: dict, *, input_locations: NativeInputLocations | None = None) -> tuple[dict, dict[str, str]]:
    if (reference.get("compression") != "gzip" or not reference.get("source_posterior_uid")
            or not isinstance(reference.get("size_bytes"), int) or reference["size_bytes"] <= 0
            or not isinstance(reference.get("uncompressed_size_bytes"), int)
            or reference["uncompressed_size_bytes"] <= 0):
        raise ValueError("Invalid complete joint source reference")
    logical = input_locations.logical_path(reference["path"]) if input_locations else resolve_repo_path(reference["path"])
    path = input_locations.file_path(logical) if input_locations else logical
    if path.stat().st_size != reference["size_bytes"]:
        raise ValueError("Complete joint source byte size changed")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != reference.get("sha256"):
        raise ValueError("Complete joint source digest changed")
    decoded = gzip.decompress(payload)
    if len(decoded) != reference["uncompressed_size_bytes"]:
        raise ValueError("Complete joint source uncompressed size changed")
    source = json.loads(decoded)
    if source.get("source_posterior_uid") != reference["source_posterior_uid"]:
        raise ValueError("Complete joint source posterior identity changed")
    return source, {str(logical): digest}
