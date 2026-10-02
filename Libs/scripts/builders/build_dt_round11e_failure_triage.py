"""Generate Round-11E analysis records from preserved Round-11D full-suite evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.evaluation.round11e_failure_triage import FROZEN_INPUT_FILENAMES  # noqa: E402
from Libs.utils.evaluation.round11e_failure_triage import build_round11e_triage  # noqa: E402
from Libs.utils.evaluation.round11e_failure_triage import validate_round11e_triage  # noqa: E402
from Libs.utils.evaluation.round11e_failure_triage import write_round11e_triage  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fail-closed Round-11E failure triage artifacts from frozen Round-11D evidence.")
    parser.add_argument("--audit-root", default="Output/audits")
    parser.add_argument("--execution", default=f"Output/audits/{FROZEN_INPUT_FILENAMES['execution']}")
    parser.add_argument("--junit", default=f"Output/audits/{FROZEN_INPUT_FILENAMES['junit']}")
    parser.add_argument("--failure-triage", default=f"Output/audits/{FROZEN_INPUT_FILENAMES['failure_triage']}")
    parser.add_argument("--closure", default=f"Output/audits/{FROZEN_INPUT_FILENAMES['closure_decision']}")
    return parser.parse_args()


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> None:
    args = parse_args()
    audit_root = repo_path(args.audit_root)
    execution_path = repo_path(args.execution)
    junit_path = repo_path(args.junit)
    failure_triage_path = repo_path(args.failure_triage)
    closure_path = repo_path(args.closure)
    payload = build_round11e_triage(REPO_ROOT, execution_path, junit_path, failure_triage_path, closure_path)
    write_round11e_triage(payload, audit_root)
    report = validate_round11e_triage(REPO_ROOT, audit_root, execution_path, junit_path, failure_triage_path, closure_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
