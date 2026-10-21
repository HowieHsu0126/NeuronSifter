"""Validate the frozen Operational Twin training protocol preflight authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.evaluation.operational_twin_training_protocol import (  # noqa: E402
    CANDIDATE_NEGATIVE_TERMINAL,
    CANDIDATE_POSITIVE_TERMINAL,
    OperationalTwinProtocolError,
    build_candidate_assignment_manifest,
    build_candidate_lineage_manifest,
    build_candidate_preflight_authority,
    load_candidate_protocol_artifacts,
    repo_path,
    validate_candidate_artifacts,
    validate_preflight_authority,
)
from Libs.scripts.builders.build_operational_twin_development_training_commands import (  # noqa: E402
    build_candidate_commands,
)
from Libs.utils.exp_utils import load_config_stack  # noqa: E402
from Libs.utils.evaluation.manuscript_native_protocol import validate_package  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "Libs/configs/config_stack.yaml")
    parser.add_argument(
        "--protocol-scope",
        choices=(
            "historical-v1",
            "manuscript-native",
            "candidate-lineage",
            "candidate-lineage-execution-interface-repair",
        ),
        default="historical-v1",
    )
    parser.add_argument("--authoritative-date", required=True)
    parser.add_argument("--output-dir", type=Path, help="Manuscript-native non-dispatch package destination")
    args = parser.parse_args()

    config = load_config_stack(args.config.resolve())
    if args.protocol_scope == "manuscript-native":
        try:
            result = validate_package(config, REPO_ROOT, args.authoritative_date, output_dir=args.output_dir)
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            result = {"status":"BLOCKED","error":str(error),"scientific_twin":"NOT_RUN",
                      "formal_qualification":False,"dispatch_authorized":False}
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "PASS" else 2

    if args.protocol_scope in {
        "candidate-lineage",
        "candidate-lineage-execution-interface-repair",
    }:
        blockers: list[dict[str, str]] = []
        successor_materialized = False
        try:
            protocol = dict(config["operational_twin_candidate_lineage_protocol"])
            successor_path = repo_path(
                REPO_ROOT,
                str(dict(protocol["output_paths"])["candidate_lineage_manifest"]),
            )
            successor_materialized = successor_path.is_file()
            if (
                args.protocol_scope
                == "candidate-lineage-execution-interface-repair"
                and not successor_materialized
            ):
                candidate_manifest = build_candidate_lineage_manifest(config)
                assignment_manifest = build_candidate_assignment_manifest(
                    config,
                    candidate_manifest,
                )
                command_catalog = build_candidate_commands(
                    config,
                    candidate_manifest,
                    assignment_manifest,
                )
                preflight = build_candidate_preflight_authority(
                    config,
                    REPO_ROOT,
                    candidate_manifest,
                    assignment_manifest,
                    command_catalog,
                )
            else:
                (
                    candidate_manifest,
                    assignment_manifest,
                    command_catalog,
                    preflight,
                ) = load_candidate_protocol_artifacts(config, REPO_ROOT)
            blockers.extend(
                validate_candidate_artifacts(
                    config,
                    REPO_ROOT,
                    candidate_manifest,
                    assignment_manifest,
                    command_catalog,
                    preflight,
                    require_attempt_roots_absent=(
                        args.protocol_scope == "candidate-lineage"
                    ),
                )
            )
            if preflight.get("authoritative_date") != args.authoritative_date:
                blockers.append(
                    {
                        "code": "candidate_authoritative_date_mismatch",
                        "message": "Candidate preflight authority date does not match CLI date.",
                    }
                )
        except (OperationalTwinProtocolError, FileNotFoundError, json.JSONDecodeError) as error:
            blockers.append(
                {
                    "code": "candidate_protocol_artifact_unavailable",
                    "message": str(error),
                }
            )
        terminal = CANDIDATE_POSITIVE_TERMINAL if not blockers else CANDIDATE_NEGATIVE_TERMINAL
        print(
            json.dumps(
                {
                    "status": "PASS" if not blockers else "BLOCKED",
                    "terminal_decision": terminal,
                    "successor_materialized": successor_materialized,
                    "blockers": blockers,
                },
                sort_keys=True,
            )
        )
        return 0 if not blockers else 2
    protocol = dict(config.get("operational_twin_training_protocol", {}))
    output_roots = dict(protocol.get("output_roots", {}))
    assignment_manifest = json.loads(repo_path(REPO_ROOT, output_roots.get("assignment_manifest", ""), must_exist=True).read_text(encoding="utf-8"))
    resource_receipt = json.loads(repo_path(REPO_ROOT, output_roots.get("resource_receipt", ""), must_exist=True).read_text(encoding="utf-8"))
    preflight = json.loads(repo_path(REPO_ROOT, output_roots.get("preflight_authority", ""), must_exist=True).read_text(encoding="utf-8"))
    blockers = validate_preflight_authority(config, REPO_ROOT, preflight, assignment_manifest, resource_receipt)
    if preflight.get("authoritative_date") != args.authoritative_date:
        blockers.append({"code": "authoritative_date_mismatch", "message": "Preflight authority date does not match CLI date."})
    status = "PASS" if not blockers else "BLOCKED"
    print(json.dumps({"status": status, "blockers": blockers}, sort_keys=True))
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
