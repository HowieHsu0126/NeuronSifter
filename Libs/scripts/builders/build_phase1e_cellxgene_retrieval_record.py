"""Bind the verified official CELLxGENE H5AD retrieval to its release evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.public_foundation_evidence import sha256_file
from Libs.utils.data.raw_data_registry import load_config, resolve_repo_path, write_json


BUILDER_ID = "Libs/scripts/builders/build_phase1e_cellxgene_retrieval_record.py"
SCHEMA_VERSION = "neuronsifter-phase1e-cellxgene-retrieval-record-v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path(value: str) -> Path:
    return resolve_repo_path(value)


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _binding(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": _relative(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _dataset_release(candidate: Mapping[str, object]) -> dict[str, object]:
    snapshot = _path(str(candidate["release_evidence_paths"][0]))
    collection = _load_json(snapshot)
    datasets = collection.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError("CELLxGENE collection snapshot has no datasets list")
    matches = [
        row for row in datasets
        if isinstance(row, dict)
        and row.get("dataset_id") == candidate["dataset_id"]
        and row.get("dataset_version_id") == candidate["source_accession"]
    ]
    if len(matches) != 1:
        raise ValueError("CELLxGENE collection snapshot does not bind exactly one configured dataset version")
    dataset = matches[0]
    assets = dataset.get("assets")
    if not isinstance(assets, list):
        raise ValueError("CELLxGENE dataset snapshot has no assets list")
    asset_matches = [
        row for row in assets
        if isinstance(row, dict)
        and row.get("filetype") == "H5AD"
        and row.get("url") == candidate["official_value_url"]
        and row.get("filesize") == candidate["expected_raw_bytes"]
    ]
    if len(asset_matches) != 1:
        raise ValueError("CELLxGENE dataset snapshot does not bind the configured H5AD asset")
    if candidate["publication_doi"] not in str(dataset.get("citation", "")):
        raise ValueError("CELLxGENE dataset citation does not bind the configured publication DOI")
    if collection.get("doi") != candidate["publication_doi"]:
        raise ValueError("CELLxGENE collection DOI does not bind the configured publication")
    return dataset


def _completed_job(job_id: str) -> dict[str, str]:
    result = subprocess.run(
        ["sacct", "-j", job_id, "--format=JobIDRaw,JobName,State,ExitCode,NodeList%20", "-P", "-n"],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [line.split("|") for line in result.stdout.splitlines() if line.strip()]
    matches = [row for row in rows if len(row) >= 5 and row[0] == job_id]
    if len(matches) != 1:
        raise ValueError(f"Could not identify one Slurm accounting row for job {job_id}")
    row = matches[0]
    if row[2] != "COMPLETED" or row[3] != "0:0":
        raise ValueError(f"CELLxGENE retrieval job is not a successful terminal job: {row[2]} {row[3]}")
    return {"job_id": row[0], "job_name": row[1], "state": row[2], "exit_code": row[3], "node": row[4]}


def _completion(log_path: Path, raw_path: Path, candidate: Mapping[str, object]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            fields[key] = value
    expected = {
        "url": str(candidate["official_value_url"]),
        "local": str(raw_path),
        "bytes": str(candidate["expected_raw_bytes"]),
        "remote_sha256": str(candidate["expected_raw_sha256"]),
        "local_sha256": str(candidate["expected_raw_sha256"]),
    }
    if any(fields.get(key) != value for key, value in expected.items()):
        raise ValueError("CELLxGENE verification log does not bind the configured official H5AD")
    if not fields.get("verified_at"):
        raise ValueError("CELLxGENE verification log has no verification timestamp")
    return fields


def build(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    config = load_config(repo_root / "Libs/configs/config_stack.yaml")
    policy = config["public_foundation_corpus"]["phase1e_cellxgene_atlas_materialization"]
    candidate = policy["candidate"]
    raw_path = _path(str(candidate["raw_value_path"]))
    raw_binding = _binding(raw_path)
    if raw_binding["bytes"] != candidate["expected_raw_bytes"] or raw_binding["sha256"] != candidate["expected_raw_sha256"]:
        raise ValueError("CELLxGENE raw asset does not equal the configured official bytes and SHA-256")
    dataset = _dataset_release(candidate)
    log_path = _path(str(candidate["retrieval_log_path"]))
    completion = _completion(log_path, raw_path, candidate)
    job = _completed_job(str(candidate["retrieval_job_id"]))
    evidence = [_binding(_path(str(value))) for value in candidate["release_evidence_paths"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "producer": BUILDER_ID,
        "created_at": _now(),
        "validation_status": "passed",
        "official_value_url": candidate["official_value_url"],
        "collection_id": candidate["collection_id"],
        "dataset_id": candidate["dataset_id"],
        "dataset_version_id": candidate["source_accession"],
        "publication_doi": candidate["publication_doi"],
        "raw_value_binding": raw_binding,
        "expected_raw_bytes": candidate["expected_raw_bytes"],
        "expected_raw_sha256": candidate["expected_raw_sha256"],
        "collection_dataset_title": dataset.get("title"),
        "collection_dataset_donor_ids": dataset.get("donor_id"),
        "official_evidence_bindings": evidence,
        "download_job": job,
        "download_log_binding": _binding(log_path),
        "download_log_completion": completion,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a CELLxGENE Phase 1E official retrieval record.")
    parser.add_argument("--output", default="Output/audits/phase1e_cellxgene_amygdala_retrieval_record.json")
    args = parser.parse_args()
    record = build(REPO_ROOT)
    write_json(_path(args.output), record)
    print(json.dumps({"status": record["validation_status"], "raw_sha256": record["raw_value_binding"]["sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
