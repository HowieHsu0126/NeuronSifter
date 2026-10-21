"""Execute one manifest-bound Operational Twin candidate attempt end to end."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.evaluation.operational_twin_candidate_execution import (  # noqa: E402
    build_and_publish_candidate_execution_gate,
    resolve_candidate_attempt_context,
    require_candidate_worker_binding,
    require_candidate_dispatch_binding,
    write_json_exclusive,
)
from Libs.utils.evaluation.operational_twin_training_protocol import (  # noqa: E402
    OperationalTwinProtocolError,
    candidate_runtime_slurm_identity,
    candidate_runtime_artifact_identity,
)
from Libs.utils.exp_utils import load_config_stack  # noqa: E402


PHASE_RECEIPT_NAME = "execution_phase_receipt.json"
PHASE_RECEIPT_SCHEMA_VERSION = (
    "neuronsifter-operational-twin-candidate-execution-phase-receipt-v2"
)


def _task_index(explicit: int | None) -> int:
    return int(explicit) if explicit is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])


def _evaluation_argv(command: Mapping[str, object]) -> list[str]:
    argv = [str(value) for value in command["argv"]]
    if len(argv) < 3 or argv[1] != "Libs/exp/training/train_operational_twin_devcal.py":
        raise OperationalTwinProtocolError(
            "candidate_attempt_command_invalid: frozen training argv has an unexpected entrypoint."
        )
    authority_flag = "--calibration-execution-authority"
    if argv.count(authority_flag) != 1:
        raise OperationalTwinProtocolError(
            "candidate_attempt_command_invalid: calibration authority must appear exactly once."
        )
    authority_index = argv.index(authority_flag)
    if authority_index + 1 >= len(argv) or not argv[authority_index + 1]:
        raise OperationalTwinProtocolError(
            "candidate_attempt_command_invalid: calibration authority path is missing."
        )
    evaluation_argv = [
        sys.executable,
        "-m",
        "Libs.exp.evaluation.evaluate_operational_twin_development",
        *argv[2:],
    ]
    if evaluation_argv.count(authority_flag) != 1:
        raise OperationalTwinProtocolError(
            "candidate_attempt_command_invalid: evaluator authority forwarding drifted."
        )
    return evaluation_argv


def _scheduler_identity() -> dict[str, object]:
    return candidate_runtime_slurm_identity(
        hostname=os.uname().nodename.split(".", 1)[0]
    )


def _phase_receipt(
    *,
    context: Mapping[str, object],
    gate: Mapping[str, object],
    started_at: str,
    phases: list[dict[str, object]],
    status: str,
    terminal_decision: str,
    error: str | None,
) -> dict[str, object]:
    execution = dict(context["execution"])
    assignment = dict(context["assignment"])
    attempt = dict(context["attempt"])
    output_root = REPO_ROOT / str(attempt["output_root"])
    evaluation_summary = output_root / "evaluation/summary.json"
    terminal_status = output_root / "terminal_status.json"
    return {
        "schema_version": PHASE_RECEIPT_SCHEMA_VERSION,
        "successor_execution_binding": dict(context["resolved_protocol"].binding),
        "runtime_dispatch_binding": gate["runtime_dispatch_binding"],
        "status": status,
        "terminal_decision": terminal_decision,
        "candidate_lineage_uid": assignment["candidate_lineage_uid"],
        "protocol_uid": assignment["protocol_uid"],
        "assignment_uid": assignment["assignment_uid"],
        "attempt_uid": attempt["attempt_uid"],
        "attempt_kind": attempt["attempt_kind"],
        "command_index": int(context["command"]["command_index"]),
        "method_id": assignment["method_id"],
        "seed": assignment["seed"],
        "output_root": attempt["output_root"],
        "source_commit": gate["source_commit"],
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "phases": phases,
        "scheduler_identity": _scheduler_identity(),
        "execution_authority": gate["execution_authority"],
        "resource_environment_receipt": gate[
            "resource_environment_receipt"
        ],
        "terminal_status": (
            candidate_runtime_artifact_identity(REPO_ROOT, terminal_status, context["resolved_protocol"])
            if terminal_status.is_file()
            else None
        ),
        "evaluation_summary": (
            candidate_runtime_artifact_identity(REPO_ROOT, evaluation_summary, context["resolved_protocol"])
            if evaluation_summary.is_file()
            else None
        ),
        "error": error,
        "access_counters": execution["access_counters"],
        "closed_flags": execution["closed_flags"],
    }


def run_attempt(task_index: int) -> int:
    config = load_config_stack(REPO_ROOT / "Libs/configs/config_stack.yaml")
    started_at = datetime.now(timezone.utc).isoformat()
    context = resolve_candidate_attempt_context(config, REPO_ROOT, task_index)
    require_candidate_worker_binding(REPO_ROOT, context["resolved_protocol"], required=True)
    require_candidate_dispatch_binding(
        config, REPO_ROOT, context["resolved_protocol"],
        assignment=context["assignment"], attempt=context["attempt"],
    )
    gate = build_and_publish_candidate_execution_gate(
        config,
        REPO_ROOT,
        task_index=task_index,
    )
    context = resolve_candidate_attempt_context(config, REPO_ROOT, task_index)
    attempt = dict(context["attempt"])
    output_root = REPO_ROOT / str(attempt["output_root"])
    phases: list[dict[str, object]] = []
    training_command = [
        sys.executable,
        "-m",
        "Libs.exp.training.run_operational_twin_assignment_command",
        "--protocol-scope",
        "candidate-lineage",
        "--task-index",
        str(task_index),
    ]
    training = subprocess.run(training_command, cwd=REPO_ROOT, check=False)
    phases.append(
        {
            "phase": "training_and_calibration",
            "argv": training_command,
            "returncode": int(training.returncode),
        }
    )
    if training.returncode != 0:
        if output_root.is_dir():
            write_json_exclusive(
                output_root / PHASE_RECEIPT_NAME,
                _phase_receipt(
                    context=context,
                    gate=gate,
                    started_at=started_at,
                    phases=phases,
                    status="FAIL",
                    terminal_decision="OPERATIONAL_TWIN_CANDIDATE_ATTEMPT_TRAINING_FAILED",
                    error=f"training runner exited {training.returncode}",
                ),
            )
        return int(training.returncode)
    evaluation_command = _evaluation_argv(context["command"])
    evaluation = subprocess.run(evaluation_command, cwd=REPO_ROOT, check=False)
    phases.append(
        {
            "phase": "open_development_calibration_evaluation",
            "argv": evaluation_command,
            "returncode": int(evaluation.returncode),
        }
    )
    if evaluation.returncode != 0:
        write_json_exclusive(
            output_root / PHASE_RECEIPT_NAME,
            _phase_receipt(
                context=context,
                gate=gate,
                started_at=started_at,
                phases=phases,
                status="FAIL",
                terminal_decision="OPERATIONAL_TWIN_CANDIDATE_ATTEMPT_EVALUATION_FAILED",
                error=f"evaluation runner exited {evaluation.returncode}",
            ),
        )
        return int(evaluation.returncode)
    write_json_exclusive(
        output_root / PHASE_RECEIPT_NAME,
        _phase_receipt(
            context=context,
            gate=gate,
            started_at=started_at,
            phases=phases,
            status="PASS",
            terminal_decision="OPERATIONAL_TWIN_CANDIDATE_ATTEMPT_EXECUTED",
            error=None,
        ),
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-index", type=int)
    args = parser.parse_args()
    return run_attempt(_task_index(args.task_index))


if __name__ == "__main__":
    raise SystemExit(main())
