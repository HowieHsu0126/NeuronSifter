"""Validate independent open-data qualification before Operational Twin training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.evaluation.operational_twin_training_feasibility import (  # noqa: E402
    qualify_canonical_open_data,
)
from Libs.utils.exp_utils import load_config_stack  # noqa: E402


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_report(path: Path, audit: dict[str, object]) -> None:
    blockers = list(dict(audit.get("retained_failure_inventory", {})).get("blockers", []))
    lines = [
        "# Controlled World Open Data Qualification",
        "",
        f"- Status: `{audit['status']}`",
        f"- Terminal: `{audit['terminal_decision']}`",
        f"- Training authorized: `{audit['training_authorized']}`",
        f"- Authoritative date: `{audit['authoritative_date']}`",
        "- Evidence boundary: controlled-world, non-empirical, open train/development/calibration only",
        "- Complete test manifest access: `0`",
        "- Sealed `twin_qualification` access: `0`",
        "- Unopened-access receipt content access: `0`",
        "",
        "## Blocking Findings",
        "",
    ]
    if blockers:
        for blocker in blockers:
            lines.append(f"- `{blocker['code']}`: {blocker['reason']}")
    else:
        lines.append("- None.")
    lines.extend([
        "",
        "## Required Plan Repair",
        "",
        "A later training round may proceed only when this authority is positive, "
        "matches the open package identities, and leaves all downstream gates closed.",
        "",
        "This audit does not establish sufficient training, Twin qualification, formal qualification, or permission to open Agent work.",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "Libs/configs/config_stack.yaml")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "Output/audits/controlled_world_open_data_qualification.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "Output/reports/controlled_world_open_data_qualification.md",
    )
    parser.add_argument("--authoritative-date", required=True)
    args = parser.parse_args()

    config = load_config_stack(args.config.resolve())
    audit = qualify_canonical_open_data(
        REPO_ROOT,
        config,
        authoritative_date=args.authoritative_date,
    )
    _write_json(args.output.resolve(), audit)
    _write_report(args.report.resolve(), audit)
    print(json.dumps({"status": audit["status"], "terminal_decision": audit["terminal_decision"]}, sort_keys=True))
    return 0 if audit["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
