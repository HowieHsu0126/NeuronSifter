"""Bind a completed official Allen WHB download to durable provenance evidence."""

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


BUILDER_ID = "Libs/scripts/builders/build_phase1e_allen_whb_retrieval_record.py"
SCHEMA_VERSION = "neuronsifter-phase1e-allen-whb-retrieval-record-v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def parse_completion_log(log_path: Path) -> dict[str, str]:
    """Read the exact completion values emitted by the download job script."""

    values: dict[str, str] = {}
    for line in log_path.read_text(encoding="utf-8", errors="strict").splitlines():
        key, separator, value = line.partition("=")
        if separator and key in {"target", "bytes", "sha256", "completed_at"}:
            values[key] = value.strip()
    required = {"target", "bytes", "sha256", "completed_at"}
    if missing := required - set(values):
        raise ValueError("Download log lacks terminal completion fields: " + ",".join(sorted(missing)))
    return values


def sacct_terminal_state(job_id: str) -> dict[str, str]:
    completed = subprocess.run(
        [
            "sacct",
            "--jobs",
            job_id,
            "--format=JobIDRaw,State,ExitCode",
            "--parsable2",
            "--noheader",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    for raw_line in completed.stdout.splitlines():
        fields = raw_line.strip().split("|")
        if len(fields) < 3 or fields[0] != str(job_id):
            continue
        state, exit_code = fields[1].strip(), fields[2].strip()
        if state != "COMPLETED" or exit_code != "0:0":
            raise ValueError(f"Download job {job_id} is not a successful terminal job: {state} {exit_code}")
        return {"job_id": str(job_id), "state": state, "exit_code": exit_code}
    raise ValueError(f"sacct has no top-level accounting row for download job {job_id}")


def build(job_id: str, job_log: Path, repo_root: Path = REPO_ROOT) -> dict[str, object]:
    config = load_config(repo_root / "Libs/configs/config_stack.yaml")
    policy = config["public_foundation_corpus"]["phase1e_atlas_materialization"]
    candidate: Mapping[str, object] = policy["candidate"]
    raw_path = resolve_repo_path(str(candidate["raw_value_path"]))
    raw = _binding(raw_path)
    expected_bytes = int(candidate["expected_raw_bytes"])
    if raw["bytes"] != expected_bytes:
        raise ValueError(f"Official WHB raw matrix has {raw['bytes']} bytes, expected {expected_bytes}")
    job_log = job_log.resolve()
    completion = parse_completion_log(job_log)
    if Path(completion["target"]).resolve() != raw_path.resolve():
        raise ValueError("Download log target does not equal configured official raw value path")
    if int(completion["bytes"]) != expected_bytes or completion["sha256"] != raw["sha256"]:
        raise ValueError("Download log completion values do not bind the current official raw matrix")
    terminal = sacct_terminal_state(str(job_id))
    return {
        "schema_version": SCHEMA_VERSION,
        "producer": BUILDER_ID,
        "created_at": _now(),
        "validation_status": "passed",
        "official_value_url": candidate["official_value_url"],
        "expression_release_version": candidate["expression_release_version"],
        "raw_value_binding": raw,
        "expected_raw_bytes": expected_bytes,
        "download_job": terminal,
        "download_log_binding": _binding(job_log),
        "download_log_completion": completion,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an evidence-bound retrieval record for the Allen WHB raw H5AD.")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--job-log", required=True)
    parser.add_argument("--output", default="Output/audits/phase1e_allen_whb_nonneurons_retrieval_record.json")
    args = parser.parse_args()
    record = build(args.job_id, Path(args.job_log), REPO_ROOT)
    output = resolve_repo_path(args.output)
    write_json(output, record)
    print(json.dumps({"status": record["validation_status"], "raw_sha256": record["raw_value_binding"]["sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
