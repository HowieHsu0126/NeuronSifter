"""Resolve the one permitted Operational Twin candidate recovery cycle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.evaluation.operational_twin_candidate_evidence import (  # noqa: E402
    build_recovery_eligibility,
    collect_scheduler_receipts,
)
from Libs.utils.evaluation.operational_twin_candidate_execution import (  # noqa: E402
    resolve_candidate_execution_protocol,
    require_clean_tracked_source,
    write_json_exclusive,
)
from Libs.utils.evaluation.operational_twin_training_protocol import (  # noqa: E402
    OperationalTwinProtocolError,
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
    args = parser.parse_args()
    config = load_config_stack(args.config.resolve())
    resolved = resolve_candidate_execution_protocol(config, REPO_ROOT)
    execution = dict(resolved.execution)
    source_commit = require_clean_tracked_source(REPO_ROOT)
    output_paths = dict(execution["output_paths"])
    dispatch_path = repo_path(
        REPO_ROOT,
        str(output_paths["dispatch_manifest"]),
        must_exist=True,
    )
    dispatch = json.loads(dispatch_path.read_text(encoding="utf-8"))
    if dispatch.get("source_commit") != source_commit:
        raise OperationalTwinProtocolError(
            "candidate_recovery_source_drift: dispatch source differs from current clean source."
        )
    scheduler = collect_scheduler_receipts(
        REPO_ROOT,
        execution,
        [dict(row) for row in dispatch["initial_arrays"]],
        phase="initial",
        source_commit=source_commit,
        resolved_context=resolved,
    )
    payload = build_recovery_eligibility(
        config,
        REPO_ROOT,
        dispatch_manifest=dispatch,
        scheduler_by_index=scheduler,
        resolved_context=resolved,
    )
    output = repo_path(REPO_ROOT, str(output_paths["recovery_eligibility"]))
    write_json_exclusive(output, payload)
    counts = {
        value: sum(
            row["recovery_eligibility"] == value for row in payload["assignments"]
        )
        for value in (
            "NOT_NEEDED",
            "ELIGIBLE",
            "NOT_ELIGIBLE_NON_INFRASTRUCTURE_FAILURE",
            "BLOCKED_INVALID_PREDECESSOR",
        )
    }
    print(json.dumps({"status": payload["status"], "counts": counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
