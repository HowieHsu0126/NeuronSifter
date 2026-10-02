"""Launch one C1 command with a versioned cluster resource profile."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config  # noqa: E402


RUN_UID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Run a C1 command through a YAML resource profile.")
    parser.add_argument("--config", default="Libs/configs/config_stack.yaml")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--run-uid", required=True)
    parser.add_argument("--root-experiment-uid", default="")
    args, command = parser.parse_known_args()
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after --")
    return args, command


def build_slurm_launch(
    config: Mapping[str, object],
    *,
    profile_id: str,
    run_uid: str,
    repo_root: Path = REPO_ROOT,
) -> tuple[list[str], dict[str, str]]:
    """Build one config-bound Slurm launch and its durable log metadata."""
    if not RUN_UID_PATTERN.fullmatch(run_uid):
        raise ValueError("C1 run UID contains unsupported characters")
    profiles = config["cluster_registry"]["foundation_state_c1"]
    if profile_id not in profiles:
        raise KeyError(f"Unknown C1 resource profile: {profile_id}")
    profile = profiles[profile_id]
    execution = config["foundation_state_c1"]["execution"]
    job_name = str(execution["job_name_prefix"]) + run_uid
    if len(job_name) > int(execution["job_name_max_length"]):
        raise ValueError("C1 Slurm job name exceeds the configured limit")

    log_directory = (repo_root / str(execution["scheduler_log_directory"])).resolve()
    log_directory.relative_to(repo_root.resolve())
    stdout_path = log_directory / f"{job_name}.stdout.log"
    stderr_path = log_directory / f"{job_name}.stderr.log"
    launch = [
        "srun",
        f"--partition={profile['partition']}",
        f"--nodelist={profile['nodelist']}",
        f"--cpus-per-task={int(profile['cpus_per_task'])}",
        f"--mem={profile['mem']}",
        f"--job-name={job_name}",
        f"--output={stdout_path.relative_to(repo_root).as_posix()}",
        f"--error={stderr_path.relative_to(repo_root).as_posix()}",
    ]
    if str(profile.get("gres") or ""):
        launch.append(f"--gres={profile['gres']}")
    metadata = {
        "job_name": job_name,
        "resource_profile": profile_id,
        "partition": str(profile["partition"]),
        "nodelist": str(profile["nodelist"]),
        "stdout_path": stdout_path.relative_to(repo_root).as_posix(),
        "stderr_path": stderr_path.relative_to(repo_root).as_posix(),
    }
    return launch, metadata


def main() -> None:
    args, command = parse_args()
    config = load_config(REPO_ROOT / args.config)
    launch, metadata = build_slurm_launch(
        config, profile_id=args.profile, run_uid=args.run_uid
    )
    for key in ("stdout_path", "stderr_path"):
        path = REPO_ROOT / metadata[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(
                f"C1 run UID must be unique; scheduler log already exists: {metadata[key]}"
            )
    environment = os.environ.copy()
    environment["NRNSIFTER_C1_RUN_UID"] = args.run_uid
    environment["NRNSIFTER_C1_RESOURCE_PROFILE"] = args.profile
    environment["NRNSIFTER_C1_ROOT_EXPERIMENT_UID"] = (
        args.root_experiment_uid or args.run_uid
    )
    environment["NRNSIFTER_C1_JOB_NAME"] = metadata["job_name"]
    environment["NRNSIFTER_C1_SCHEDULER_STDOUT_PATH"] = metadata["stdout_path"]
    environment["NRNSIFTER_C1_SCHEDULER_STDERR_PATH"] = metadata["stderr_path"]
    print(json.dumps({"command": command, **metadata}, sort_keys=True), flush=True)
    runtime_wrapper = [
        "bash",
        "-c",
        (
            "printf 'c1_slurm_job_id=%s\\n' \"${SLURM_JOB_ID:-unavailable}\"; "
            "printf 'c1_slurm_job_name=%s\\n' \"${SLURM_JOB_NAME:-unavailable}\"; "
            "printf 'c1_slurm_node=%s\\n' \"${SLURMD_NODENAME:-$(hostname)}\"; "
            "printf 'c1_started_at=%s\\n' \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"; "
            "\"$@\"; status=$?; "
            "printf 'c1_ended_at=%s\\n' \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"; "
            "printf 'c1_exit_code=%s\\n' \"$status\"; exit \"$status\""
        ),
        "c1-runtime",
        *command,
    ]
    completed = subprocess.run(
        [*launch, *runtime_wrapper],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
    )
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
