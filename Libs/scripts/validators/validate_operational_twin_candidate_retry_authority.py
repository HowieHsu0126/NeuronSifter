"""Validate prepublication or materialized Operational Twin retry authority."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.evaluation.operational_twin_candidate_retry_authority import (  # noqa: E402
    POSITIVE,
    adjudicate_retry_authority,
    render_positive_successor_report,
    render_retry_authority_report,
    validate_positive_successor_package,
    validate_retry_authority_config,
)
from Libs.utils.evaluation.operational_twin_training_protocol import (  # noqa: E402
    OperationalTwinProtocolError,
)
from Libs.utils.exp_utils import load_config_stack  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "Libs/configs/config_stack.yaml")
    parser.add_argument("--mode", choices=("adjudicate", "materialized"), required=True)
    args = parser.parse_args()
    config = load_config_stack(args.config.resolve())
    blockers = validate_retry_authority_config(config)
    audit: dict[str, object] | None = None
    try:
        audit = adjudicate_retry_authority(config, REPO_ROOT)
        if args.mode == "materialized":
            contract = dict(config["operational_twin_candidate_retry_authority"])
            paths = dict(contract["output_paths"])
            audit_path = REPO_ROOT / str(paths["decision_audit"])
            report_path = REPO_ROOT / str(paths["decision_report"])
            expected_audit_text = json.dumps(audit, indent=2, sort_keys=True) + "\n"
            if (
                not audit_path.is_file()
                or audit_path.read_text(encoding="utf-8") != expected_audit_text
            ):
                blockers.append({"code": "retry_authority_audit_mismatch", "message": "materialized decision audit differs from recomputation"})
            if not report_path.is_file() or report_path.read_text(encoding="utf-8") != render_retry_authority_report(audit):
                blockers.append({"code": "retry_authority_report_mismatch", "message": "materialized decision report differs from recomputation"})
            positive_paths = [REPO_ROOT / str(paths[key]) for key in ("candidate_lineage_manifest", "assignment_manifest", "assignment_commands", "preflight_authority", "preflight_report")]
            if audit["terminal_decision"] == POSITIVE:
                package: dict[str, object] = {}
                for key, path in zip(
                    (
                        "candidate_lineage_manifest",
                        "assignment_manifest",
                        "assignment_commands",
                        "preflight_authority",
                    ),
                    positive_paths[:4],
                    strict=True,
                ):
                    if not path.is_file():
                        blockers.append(
                            {
                                "code": "retry_authority_positive_artifact_missing",
                                "message": key,
                            }
                        )
                        continue
                    text = path.read_text(encoding="utf-8")
                    payload = json.loads(text)
                    if not isinstance(payload, dict) or text != (
                        json.dumps(payload, indent=2, sort_keys=True) + "\n"
                    ):
                        blockers.append(
                            {
                                "code": "retry_authority_positive_artifact_encoding_invalid",
                                "message": key,
                            }
                        )
                    package[key] = payload
                if not positive_paths[4].is_file():
                    blockers.append({"code": "retry_authority_positive_report_missing", "message": str(positive_paths[4])})
                else:
                    package["preflight_report"] = positive_paths[4].read_text(
                        encoding="utf-8"
                    )
                if len(package) == 5:
                    blockers.extend(
                        validate_positive_successor_package(
                            config,
                            REPO_ROOT,
                            audit,
                            package,
                            expected_audit_materialized=True,
                            require_publication_targets_absent=False,
                        )
                    )
                    preflight = package["preflight_authority"]
                    if (
                        isinstance(preflight, dict)
                        and package["preflight_report"]
                        != render_positive_successor_report(preflight)
                    ):
                        blockers.append(
                            {
                                "code": "retry_authority_positive_report_mismatch",
                                "message": str(positive_paths[4]),
                            }
                        )
            elif any(os.path.lexists(path) for path in positive_paths):
                blockers.append({"code": "retry_authority_negative_branch_leak", "message": "positive-only artifacts exist for the successor-required branch"})
    except (OperationalTwinProtocolError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        blockers.append({"code": "retry_authority_authentication_failed", "message": str(error)})
    payload = {
        "status": "PASS" if not blockers else "BLOCKED",
        "terminal_decision": audit.get("terminal_decision") if audit else None,
        "next_direction": audit.get("next_direction") if audit else None,
        "assignment_count": audit.get("assignment_count") if audit else 0,
        "authorized_row_count": audit.get("authorized_row_count") if audit else 0,
        "successor_required_row_count": audit.get("successor_required_row_count") if audit else 0,
        "candidate_dispatch_authorized": False,
        "side_effect_count": 0,
        "blockers": blockers,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if not blockers else 2


if __name__ == "__main__":
    raise SystemExit(main())
