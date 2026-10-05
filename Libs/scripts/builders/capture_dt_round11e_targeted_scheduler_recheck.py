"""Persist a non-mutating scheduler recheck for Round-11E targeted jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import write_json, write_text_atomic  # noqa: E402


TARGETED_JOB_IDS = ("490950", "490951", "490952", "490953")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str]) -> dict[str, object]:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _write_stream_pair(audit_root: Path, prefix: str, result: dict[str, object]) -> dict[str, object]:
    stdout_path = audit_root / f"{prefix}.stdout"
    stderr_path = audit_root / f"{prefix}.stderr"
    write_text_atomic(stdout_path, str(result["stdout"]))
    write_text_atomic(stderr_path, str(result["stderr"]))
    return {
        "command": result["command"],
        "returncode": result["returncode"],
        "stdout": {
            "path": stdout_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": _sha256_file(stdout_path),
            "bytes": stdout_path.stat().st_size,
        },
        "stderr": {
            "path": stderr_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": _sha256_file(stderr_path),
            "bytes": stderr_path.stat().st_size,
        },
    }


def _capture_scontrol(audit_root: Path) -> dict[str, object]:
    per_job: dict[str, object] = {}
    for job_id in TARGETED_JOB_IDS:
        result = _run(["scontrol", "show", "job", job_id])
        per_job[job_id] = _write_stream_pair(
            audit_root,
            f"dt_round11e_targeted_scheduler_recheck_scontrol_{job_id}",
            result,
        )
    return {
        "per_job": per_job,
        "all_job_queries_succeeded": all(
            _as_mapping(result, f"scontrol.{job_id}").get("returncode") == 0
            for job_id, result in per_job.items()
        ),
    }


def _as_mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping for {label}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture non-mutating scheduler evidence for Round-11E targeted jobs.")
    parser.add_argument("--audit-root", default="Output/audits")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit_root = Path(args.audit_root)
    if not audit_root.is_absolute():
        audit_root = REPO_ROOT / audit_root
    scontrol = _capture_scontrol(audit_root)
    sacct_result = _run(
        [
            "sacct",
            "-j",
            ",".join(TARGETED_JOB_IDS),
            "--format=JobIDRaw,JobName%40,State,ExitCode,NodeList%20,Start,End",
            "-P",
            "-n",
        ]
    )
    sacct = _write_stream_pair(audit_root, "dt_round11e_targeted_scheduler_recheck_sacct", sacct_result)
    scontrol_errors = []
    for result in _as_mapping(scontrol.get("per_job"), "scontrol.per_job").values():
        stderr = _as_mapping(result, "scontrol.result").get("stderr")
        stream = _as_mapping(stderr, "scontrol.result.stderr")
        stderr_path = REPO_ROOT / str(stream.get("path"))
        scontrol_errors.append(stderr_path.read_text(encoding="utf-8"))
    scontrol_error = "\n".join(scontrol_errors)
    sacct_error = str(sacct_result["stderr"])
    payload = {
        "schema_version": "dt-round11e-targeted-scheduler-recheck-v1",
        "audit_kind": "round11e_targeted_scheduler_recheck",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "job_ids": list(TARGETED_JOB_IDS),
        "scontrol": scontrol,
        "sacct": sacct,
        "scontrol_available": bool(scontrol.get("all_job_queries_succeeded")) and "Invalid job id" not in scontrol_error,
        "sacct_available": sacct_result["returncode"] == 0 and not sacct_error.strip(),
        "conclusion": {
            "scontrol_status": "expired_or_unavailable"
            if "Invalid job id" in scontrol_error
            else "response_captured_without_terminal_assertion",
            "sacct_status": "unavailable"
            if sacct_result["returncode"] != 0 or sacct_error.strip()
            else "response_captured_without_terminal_assertion",
            "terminal_evidence_scope": "This post-retention recheck does not replace preserved JUnit/stdout/provenance evidence or assert a new terminal state.",
        },
    }
    output_path = audit_root / "dt_round11e_targeted_scheduler_recheck.json"
    write_json(output_path, payload)
    print(
        json.dumps(
            {
                "status": "captured",
                "output": output_path.relative_to(REPO_ROOT).as_posix(),
                "scontrol_available": payload["scontrol_available"],
                "sacct_available": payload["sacct_available"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
