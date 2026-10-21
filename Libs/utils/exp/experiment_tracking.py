from __future__ import annotations

from datetime import datetime
import hashlib
import os
from pathlib import Path
import platform
import random
import socket
import subprocess
import sys

import numpy as np
import torch

from Libs.utils.exp.logger import get_pipeline_logger
from Libs.utils.data.raw_data_registry import repo_root, write_json, write_text_atomic


def timestamp_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log_event(message: str, **fields: object) -> None:
    get_pipeline_logger().progress(message, **fields)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def scoped_seed(base_seed: int, *parts: str) -> int:
    digest = hashlib.sha1("::".join([str(base_seed), *parts]).encode("utf-8")).hexdigest()[:8]
    return int(base_seed) + (int(digest, 16) % 1_000_000)


def command_output(command: list[str], working_directory: Path) -> str:
    completed = subprocess.run(
        command,
        cwd=working_directory,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def capture_run_metadata(
    artifact_root: Path,
    config_path: Path,
    argv: list[str],
    seed: int,
    extra_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    repository_root = repo_root()
    config_snapshot_path = artifact_root / "config_snapshot.yaml"
    write_text_atomic(config_snapshot_path, config_path.read_text(encoding="utf-8"))

    git_hash = command_output(["git", "rev-parse", "HEAD"], repository_root)
    git_status = command_output(["git", "status", "--short"], repository_root)
    git_diff = subprocess.run(
        ["git", "diff", "--binary"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    git_diff_path = artifact_root / "uncommitted.diff"
    write_text_atomic(git_diff_path, git_diff)

    pip_freeze = command_output([sys.executable, "-m", "pip", "freeze"], repository_root)
    pip_freeze_path = artifact_root / "pip_freeze.txt"
    write_text_atomic(pip_freeze_path, pip_freeze + "\n")

    payload = {
        "timestamp": timestamp_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "seed": int(seed),
        "command": " ".join(argv),
        "config_path": str(config_path),
        "config_snapshot_path": str(config_snapshot_path.relative_to(repository_root)),
        "git_hash": git_hash,
        "git_status": git_status.splitlines(),
        "git_diff_path": str(git_diff_path.relative_to(repository_root)),
        "pip_freeze_path": str(pip_freeze_path.relative_to(repository_root)),
        "extra_metadata": extra_metadata or {},
    }
    metadata_path = artifact_root / "run_metadata.json"
    write_json(metadata_path, payload)
    return payload
