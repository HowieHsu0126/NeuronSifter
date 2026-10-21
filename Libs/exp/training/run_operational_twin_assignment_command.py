"""Run one manifest-bound Operational Twin assignment command by catalog index."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.evaluation.operational_twin_training_protocol import (  # noqa: E402
    OperationalTwinProtocolError,
    load_candidate_protocol_artifacts,
    repo_path,
    require_candidate_assignment_preflight,
    require_candidate_execution_authority,
    validate_candidate_artifacts,
)
from Libs.utils.evaluation.operational_twin_candidate_execution import (  # noqa: E402
    resolve_candidate_execution_protocol,
)
from Libs.utils.evaluation.operational_twin_successor_execution_handoff import (  # noqa: E402
    _path as handoff_path,
)
from Libs.utils.exp_utils import load_config_stack  # noqa: E402


def _argv_value(argv: list[str], flag: str) -> str:
    if flag not in argv:
        raise OperationalTwinProtocolError(f"candidate_command_flag_missing: {flag}")
    index = argv.index(flag)
    if index + 1 >= len(argv):
        raise OperationalTwinProtocolError(f"candidate_command_flag_value_missing: {flag}")
    return argv[index + 1]


def _candidate_catalog_path(explicit_path: Path | None) -> Path:
    config = load_config_stack(REPO_ROOT / "Libs/configs/config_stack.yaml")
    resolved = resolve_candidate_execution_protocol(config, REPO_ROOT)
    relative = str(resolved.protocol["output_paths"]["assignment_commands"])
    canonical = handoff_path(REPO_ROOT, relative)
    if explicit_path is not None:
        supplied = explicit_path if explicit_path.is_absolute() else REPO_ROOT / explicit_path
        supplied_relative = supplied.relative_to(REPO_ROOT).as_posix()
        if handoff_path(REPO_ROOT, supplied_relative) != canonical:
            raise OperationalTwinProtocolError("candidate_command_catalog_path_mismatch")
    return canonical


def _run_candidate_command(catalog_path: Path, task_index: int) -> int:
    config = load_config_stack(REPO_ROOT / "Libs/configs/config_stack.yaml")
    resolved = resolve_candidate_execution_protocol(config, REPO_ROOT)
    protocol = dict(resolved.protocol)
    canonical_catalog = repo_path(
        REPO_ROOT,
        dict(protocol["output_paths"])["assignment_commands"],
        must_exist=True,
    )
    supplied_catalog = catalog_path if catalog_path.is_absolute() else REPO_ROOT / catalog_path
    if handoff_path(REPO_ROOT, supplied_catalog.relative_to(REPO_ROOT).as_posix()) != canonical_catalog:
        raise OperationalTwinProtocolError(
            "candidate_command_catalog_path_mismatch: only the frozen canonical catalog may execute."
        )
    candidate_manifest, assignment_manifest, catalog, preflight = load_candidate_protocol_artifacts(
        config, REPO_ROOT, resolved_context=resolved
    )
    blockers = validate_candidate_artifacts(
        config,
        REPO_ROOT,
        candidate_manifest,
        assignment_manifest,
        catalog,
        preflight,
        require_attempt_roots_absent=False,
        resolved_context=resolved,
    )
    if blockers:
        raise OperationalTwinProtocolError(
            "; ".join(f"{row['code']}: {row['message']}" for row in blockers)
        )
    commands = list(catalog["commands"])
    if task_index < 0 or task_index >= len(commands):
        raise OperationalTwinProtocolError(
            f"candidate_task_index_out_of_bounds: {task_index} is outside [0, {len(commands) - 1}]."
        )
    row = dict(commands[task_index])
    argv = list(row["argv"])
    candidate_context = require_candidate_assignment_preflight(
        config,
        REPO_ROOT,
        candidate_lineage_uid=str(row["candidate_lineage_uid"]),
        protocol_uid=str(row["protocol_uid"]),
        assignment_uid=str(row["assignment_uid"]),
        attempt_uid=str(row["attempt_uid"]),
        method_id=str(row["method_id"]),
        seed=int(row["seed"]),
        output_root=str(row["output_root"]),
        candidate_manifest_hash=str(catalog["candidate_manifest_hash"]),
        assignment_manifest_hash=str(catalog["assignment_manifest_hash"]),
        command_catalog_hash_value=str(catalog["command_catalog_hash"]),
        preflight_uid=str(catalog["preflight_uid"]),
        state_inputs={field: None for field in (
            "resume_input",
            "history_input",
            "checkpoint_input",
            "optimizer_state_input",
            "model_state_input",
            "calibration_state_input",
        )},
        require_recovery_predecessor=True,
        resolved_context=resolved,
    )
    require_candidate_execution_authority(
        config,
        REPO_ROOT,
        assignment=candidate_context["assignment"],
        attempt=candidate_context["attempt"],
        candidate_manifest=candidate_context["candidate_manifest"],
        assignment_manifest=candidate_context["assignment_manifest"],
        command_catalog=candidate_context["command_catalog"],
        preflight=candidate_context["preflight"],
        execution_authority_path=str(row["execution_authority"]),
        resource_environment_receipt_path=str(
            row["resource_environment_receipt"]
        ),
        cuda_available=bool(torch.cuda.is_available()),
        cuda_device_count=int(torch.cuda.device_count()),
        torch_version=str(torch.__version__),
        resolved_context=resolved,
    )
    if _argv_value(argv, "--command-catalog-hash") != catalog["command_catalog_hash"]:
        raise OperationalTwinProtocolError(
            "candidate_command_catalog_hash_argument_mismatch: stored argv does not bind the catalog hash."
        )
    print(
        json.dumps(
            {
                "selected_assignment": row["assignment_uid"],
                "selected_attempt": row["attempt_uid"],
                "candidate_lineage_uid": row["candidate_lineage_uid"],
                "protocol_uid": row["protocol_uid"],
                "command_catalog_hash": catalog["command_catalog_hash"],
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                "command_git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
                ).strip(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return subprocess.run(argv, cwd=REPO_ROOT, shell=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol-scope",
        choices=("historical-v1", "candidate-lineage"),
        default="historical-v1",
    )
    parser.add_argument("--command-catalog", type=Path)
    parser.add_argument("--task-index", type=int, default=None)
    args = parser.parse_args()
    task_index = args.task_index
    if task_index is None:
        task_index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if args.protocol_scope == "candidate-lineage":
        catalog_path = _candidate_catalog_path(args.command_catalog)
        return _run_candidate_command(catalog_path, task_index)
    catalog_path = args.command_catalog or (
        REPO_ROOT
        / "Output/results/operational_twin/development_training/catalog/assignment_commands.json"
    )
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if task_index < 0 or task_index >= len(catalog["commands"]):
        raise IndexError(f"task index {task_index} is outside the command catalog")
    row = catalog["commands"][task_index]
    print(json.dumps({
        "selected_assignment": row["assignment_uid"],
        "method_id": row["method_id"],
        "seed": row["seed"],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "command_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
    }, sort_keys=True), flush=True)
    command = row["command"].replace(" --dry-run", "")
    return subprocess.run(shlex.split(command), cwd=REPO_ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
