"""Record clean-worktree provenance for the Round-11E repair gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config_with_provenance, write_json


LOCK_PATTERNS = (
    "requirements*.txt",
    "environment*.yaml",
    "environment*.yml",
    "pyproject.toml",
    "poetry.lock",
    "uv.lock",
    "conda-lock.*",
    "Pipfile.lock",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(*command: str, cwd: Path | None = None) -> str:
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _dependency_lock_record(repository_root: Path) -> dict[str, object]:
    paths: list[Path] = []
    for pattern in LOCK_PATTERNS:
        paths.extend(path for path in repository_root.glob(pattern) if path.is_file())
    unique_paths = sorted(set(paths))
    if not unique_paths:
        return {"status": "not_available", "sha256": None, "files": []}
    file_rows = [
        {
            "path": path.relative_to(repository_root).as_posix(),
            "sha256": _sha256_file(path),
        }
        for path in unique_paths
    ]
    return {"status": "available", "sha256": _canonical_sha256(file_rows), "files": file_rows}


def _environment_record() -> dict[str, object]:
    pip_freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=False,
        capture_output=True,
        text=True,
    )
    normalized_freeze = "\n".join(sorted(line for line in pip_freeze.stdout.splitlines() if line.strip()))
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "pip_freeze_exit_code": pip_freeze.returncode,
        "pip_freeze_sha256": hashlib.sha256(normalized_freeze.encode("utf-8")).hexdigest(),
        "environment_hash": hashlib.sha256(
            (sys.executable + "\n" + sys.version + "\n" + normalized_freeze).encode("utf-8")
        ).hexdigest(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Round-11E clean-worktree provenance.")
    parser.add_argument("--repository-root", required=True)
    parser.add_argument("--audit-root", default="Output/audits")
    parser.add_argument(
        "--output",
        default="dt_round11e_clean_worktree_provenance.json",
        help="Output filename or absolute path; defaults to the canonical Round-11E provenance record.",
    )
    parser.add_argument(
        "--frozen-execution",
        default="Output/audits/dt_round11d_full_suite_execution.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repository_root = Path(args.repository_root).resolve()
    audit_root = Path(args.audit_root).resolve()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = audit_root / output_path
    frozen_execution_path = Path(args.frozen_execution).resolve()
    config_path = repository_root / "Libs/configs/config_stack.yaml"
    config, config_provenance = load_config_with_provenance(config_path)
    dirty_paths = _run("git", "status", "--porcelain=v1", "-uno", cwd=repository_root).splitlines()
    commit = _run("git", "rev-parse", "HEAD", cwd=repository_root)
    frozen_execution = json.loads(frozen_execution_path.read_text(encoding="utf-8"))
    payload = {
        "schema_version": "dt-round11e-clean-worktree-provenance-v1",
        "audit_kind": "clean_repair_worktree_provenance",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "repository_root": str(repository_root),
        "git_commit": commit,
        "dirty_status": "clean" if not dirty_paths else "dirty",
        "dirty_paths": dirty_paths,
        "sparse_checkout": {
            "enabled": _run("git", "config", "--worktree", "--get", "core.sparseCheckout", cwd=repository_root) == "true",
            "patterns": _run("git", "sparse-checkout", "list", cwd=repository_root).splitlines(),
        },
        "config": {
            "path": "Libs/configs/config_stack.yaml",
            "sha256": _sha256_file(config_path),
            "resolved_config_sha256": _canonical_sha256(config),
            "resolved_schema_registry_sha256": _canonical_sha256(config.get("schemas", {})),
            "include_order": config_provenance["include_order"],
        },
        "environment": _environment_record(),
        "dependency_lock": _dependency_lock_record(repository_root),
        "host_resource_profile": {
            "hostname": platform.node(),
            "cpu_count": os.cpu_count(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "slurm_nodelist": os.environ.get("SLURMD_NODENAME"),
        },
        "frozen_job_relation": {
            "job_id": frozen_execution.get("job_id"),
            "original_commit": frozen_execution.get("clean_worktree_commit"),
            "original_config_sha256": frozen_execution.get("clean_config_sha256"),
            "original_environment_hash": frozen_execution.get("environment_hash"),
            "reproduction_relation": "exact_commit_and_config_sparse_reconstruction",
            "commit_matches": commit == frozen_execution.get("clean_worktree_commit"),
            "config_matches": _sha256_file(config_path) == frozen_execution.get("clean_config_sha256"),
            "frozen_execution_sha256": _sha256_file(frozen_execution_path),
        },
        "known_provenance_limitations": [
            "The original job did not capture clean-worktree dirty status at submission or terminal time.",
            "Input/processed artifacts are ignored by Git and intentionally absent from this sparse clean worktree.",
            "This artifact establishes exact code/config identity, not identical generated-input availability or dependency-lock identity.",
        ],
    }
    if payload["dirty_status"] != "clean":
        raise RuntimeError("Round-11E repair worktree must be clean before execution")
    if not payload["frozen_job_relation"]["commit_matches"] or not payload["frozen_job_relation"]["config_matches"]:
        raise RuntimeError("Clean repair worktree does not match the frozen job code/config identity")
    write_json(output_path, payload)
    print(json.dumps({"status": "passed", "commit": commit, "dirty_status": payload["dirty_status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
