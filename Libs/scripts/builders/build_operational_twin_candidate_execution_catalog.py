"""Build the final 15-assignment/30-slot Operational Twin execution package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.evaluation.operational_twin_candidate_evidence import (  # noqa: E402
    artifact_identity,
    build_execution_audit,
    build_execution_catalog,
    merge_scheduler_evidence,
    load_scheduler_receipts,
    render_execution_report,
    validate_execution_catalog,
)
from Libs.utils.evaluation.operational_twin_candidate_execution import (  # noqa: E402
    resolve_candidate_execution_protocol,
    require_clean_tracked_source,
    write_bytes_exclusive,
)
from Libs.utils.evaluation.operational_twin_training_protocol import (  # noqa: E402
    OperationalTwinProtocolError,
    repo_path,
)
from Libs.utils.exp_utils import load_config_stack  # noqa: E402


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "Libs/configs/config_stack.yaml",
    )
    args = parser.parse_args()
    config = load_config_stack(args.config.resolve())
    resolved = resolve_candidate_execution_protocol(config, REPO_ROOT)
    execution = dict(resolved.execution)
    require_clean_tracked_source(REPO_ROOT)
    output_paths = dict(execution["output_paths"])
    dispatch_path = repo_path(
        REPO_ROOT,
        str(output_paths["dispatch_manifest"]),
        must_exist=True,
    )
    recovery_path = repo_path(
        REPO_ROOT,
        str(output_paths["recovery_eligibility"]),
        must_exist=True,
    )
    dispatch = json.loads(dispatch_path.read_text(encoding="utf-8"))
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    source_commit = dispatch["source_commit"]
    if recovery.get("source_commit") != source_commit:
        raise OperationalTwinProtocolError(
            "candidate_execution_source_drift: recovery evidence differs from its initial dispatch."
        )
    scheduler = load_scheduler_receipts(
        REPO_ROOT,
        execution,
        [dict(row) for row in dispatch["initial_arrays"]],
        phase="initial",
        source_commit=source_commit,
        resolved_context=resolved,
    )
    recovery_dispatch_path = repo_path(REPO_ROOT, str(output_paths["recovery_dispatch_manifest"]))
    recovery_dispatch = (
        json.loads(recovery_dispatch_path.read_text()) if recovery_dispatch_path.is_file() else None
    )
    recovery_arrays = [dict(row) for row in (recovery_dispatch or {}).get("recovery_arrays", [])]
    if recovery_arrays:
        scheduler = merge_scheduler_evidence(
            scheduler, load_scheduler_receipts(
                REPO_ROOT,
                execution,
                recovery_arrays,
                phase="recovery",
                source_commit=recovery_dispatch["source_commit"],
                resolved_context=resolved,
            )
        )
    catalog = build_execution_catalog(
        config,
        REPO_ROOT,
        dispatch_manifest=dispatch,
        recovery_eligibility=recovery,
        scheduler_by_index=scheduler,
        resolved_context=resolved,
    )
    catalog["dispatch_manifest"] = artifact_identity(REPO_ROOT, dispatch_path)
    catalog["recovery_dispatch_manifest"] = (
        artifact_identity(REPO_ROOT, recovery_dispatch_path) if recovery_dispatch is not None else None
    )
    catalog["recovery_eligibility"] = artifact_identity(REPO_ROOT, recovery_path)
    catalog_blockers = validate_execution_catalog(config, REPO_ROOT, catalog, resolved_context=resolved)
    if catalog_blockers:
        catalog["status"] = "FAIL"
        catalog["validation_blockers"].extend(catalog_blockers)
        if catalog["terminal_decision"] == execution["terminal_policy"]["positive"]:
            catalog["terminal_decision"] = execution["terminal_policy"]["invalid"]
    audit = build_execution_audit(catalog)
    report = render_execution_report(catalog, audit).encode()
    catalog_path = repo_path(
        REPO_ROOT,
        str(output_paths["candidate_execution_catalog"]),
    )
    audit_path = repo_path(REPO_ROOT, str(output_paths["execution_audit"]))
    report_path = repo_path(REPO_ROOT, str(output_paths["execution_report"]))
    write_bytes_exclusive(catalog_path, _json_bytes(catalog))
    write_bytes_exclusive(audit_path, _json_bytes(audit))
    write_bytes_exclusive(report_path, report)
    print(
        json.dumps(
            {
                "status": catalog["status"],
                "terminal_decision": catalog["terminal_decision"],
                "assignment_count": catalog["assignment_count"],
                "attempt_slot_count": catalog["attempt_slot_count"],
                "validation_blocker_count": len(catalog["validation_blockers"]),
            },
            sort_keys=True,
        )
    )
    return 0 if catalog["terminal_decision"] == execution["terminal_policy"]["positive"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
