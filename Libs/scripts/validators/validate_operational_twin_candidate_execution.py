"""Validate the Operational Twin candidate execution contract or one live gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.evaluation.operational_twin_candidate_execution import (  # noqa: E402
    build_candidate_pre_dispatch_audit,
    build_live_candidate_resource_probe,
    validate_candidate_execution_config,
    validate_candidate_execution_interface_repair,
    validate_candidate_receipt_semantics_amendment,
    validate_published_candidate_execution_gate,
    write_json_exclusive,
)
from Libs.utils.evaluation.operational_twin_candidate_evidence import (  # noqa: E402
    require_recovery_eligibility,
    validate_scheduler_accounting_fixture,
    validate_execution_catalog,
)
from Libs.utils.evaluation.operational_twin_training_protocol import (  # noqa: E402
    repo_path,
)
from Libs.utils.exp_utils import load_config_stack  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "Libs/configs/config_stack.yaml",
    )
    parser.add_argument(
        "--mode",
        choices=(
            "config",
            "execution-interface-repair",
            "receipt-semantics-amendment",
            "live-resource-probe",
            "pre-dispatch",
            "attempt-gate",
            "recovery",
            "final",
        ),
        default="config",
    )
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument(
        "--resource-profile",
        choices=("cpu_light", "gpu_single"),
    )
    args = parser.parse_args()
    config = load_config_stack(args.config.resolve())
    if args.mode == "config":
        blockers = validate_candidate_execution_config(config, REPO_ROOT)
        payload: dict[str, object] = {
            "status": "PASS" if not blockers else "FAIL",
            "blocker_count": len(blockers),
            "blockers": blockers,
        }
        print(json.dumps(payload, sort_keys=True))
        return 0 if not blockers else 1
    if args.mode == "execution-interface-repair":
        protocol = dict(config["operational_twin_candidate_lineage_protocol"])
        protocol_paths = dict(protocol["output_paths"])
        materialized = repo_path(
            REPO_ROOT,
            str(protocol_paths["candidate_lineage_manifest"]),
        ).is_file()
        blockers = validate_candidate_execution_interface_repair(
            config,
            REPO_ROOT,
            require_materialized_artifacts=materialized,
        )
        try:
            source_bindings = dict(protocol["source_bindings"])
            fixture_path = repo_path(
                REPO_ROOT,
                str(source_bindings["array_terminal_accounting_fixture"]),
                must_exist=True,
            )
            fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
            blockers.extend(validate_scheduler_accounting_fixture(fixture))
        except (KeyError, FileNotFoundError, json.JSONDecodeError) as error:
            blockers.append(
                {
                    "code": "candidate_execution_interface_fixture_invalid",
                    "message": str(error),
                }
            )
        repair = dict(config["operational_twin_candidate_execution"])[
            "interface_repair_contract"
        ]
        payload = {
            "status": "PASS" if not blockers else "FAIL",
            "terminal_decision": (
                repair["positive_terminal"]
                if not blockers
                else repair["negative_terminal"]
            ),
            "blocker_count": len(blockers),
            "blockers": blockers,
            "successor_materialized": materialized,
            "candidate_dispatch_authorized": False,
            "side_effect_count": 0,
        }
        print(json.dumps(payload, sort_keys=True))
        return 0 if not blockers else 1
    if args.mode == "receipt-semantics-amendment":
        blockers = validate_candidate_receipt_semantics_amendment(
            config,
            REPO_ROOT,
        )
        payload = {
            "status": "PASS" if not blockers else "FAIL",
            "terminal_decision": (
                "OPERATIONAL_TWIN_RESOURCE_RECEIPT_SEMANTICS_FROZEN"
                if not blockers
                else "OPERATIONAL_TWIN_RESOURCE_RECEIPT_SEMANTICS_INVALID"
            ),
            "blocker_count": len(blockers),
            "blockers": blockers,
        }
        print(json.dumps(payload, sort_keys=True))
        return 0 if not blockers else 1
    if args.mode == "live-resource-probe":
        if args.resource_profile is None or args.artifact is None:
            parser.error(
                "live-resource-probe requires --resource-profile and --artifact"
            )
        payload = build_live_candidate_resource_probe(
            config,
            REPO_ROOT,
            resource_profile=args.resource_profile,
        )
        write_json_exclusive(args.artifact.resolve(), payload)
        print(json.dumps(payload, sort_keys=True))
        return 0
    if args.mode == "pre-dispatch":
        payload = build_candidate_pre_dispatch_audit(config, REPO_ROOT)
        execution = dict(config["operational_twin_candidate_execution"])
        output = args.artifact or repo_path(
            REPO_ROOT,
            str(dict(execution["output_paths"])["pre_dispatch_audit"]),
        )
        write_json_exclusive(output, payload)
        print(json.dumps(payload, sort_keys=True))
        return 0 if payload["status"] == "PASS" else 1
    if args.mode == "recovery":
        execution = config["operational_twin_candidate_execution"]
        artifact = args.artifact or repo_path(
            REPO_ROOT, execution["output_paths"]["recovery_eligibility"]
        )
        payload = require_recovery_eligibility(config, REPO_ROOT, artifact)
        print(json.dumps({"status": payload["status"], "assignment_count": payload["assignment_count"]}, sort_keys=True))
        return 0 if payload["status"] == "PASS" else 1
    if args.mode == "final":
        execution = dict(config["operational_twin_candidate_execution"])
        artifact = args.artifact or repo_path(
            REPO_ROOT,
            str(dict(execution["output_paths"])["candidate_execution_catalog"]),
        )
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        blockers = validate_execution_catalog(config, REPO_ROOT, payload)
        print(json.dumps({"status": "PASS" if not blockers else "FAIL", "blockers": blockers}, sort_keys=True))
        return 0 if not blockers else 1
    task_index = (
        int(args.task_index)
        if args.task_index is not None
        else int(os.environ["SLURM_ARRAY_TASK_ID"])
    )
    result = validate_published_candidate_execution_gate(
        config,
        REPO_ROOT,
        task_index=task_index,
    )
    print(json.dumps({"status": "PASS", **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
