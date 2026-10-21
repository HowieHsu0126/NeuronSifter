#!/usr/bin/env python3
"""Validate Round-20 Allen official metadata resolution artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.round20_allen_official_metadata_resolution import (
    CONFIG_KEY,
    CONFIG_RELATIVE,
    validate_contract_definition,
    validate_emitted_artifacts,
)


def load_contract(repository: Path, config: Path) -> dict[str, object]:
    config_path = config if config.is_absolute() else repository / config
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get(CONFIG_KEY), dict):
        raise ValueError(f"{CONFIG_KEY} mapping is required")
    return dict(payload[CONFIG_KEY])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--config", type=Path, default=CONFIG_RELATIVE)
    parser.add_argument("--contract-only", action="store_true")
    args = parser.parse_args()
    repository = args.repository_root.resolve()
    contract = load_contract(repository, args.config)
    blockers = validate_contract_definition(contract) if args.contract_only else validate_emitted_artifacts(contract, repository=repository)
    status = "passed" if not blockers else "blocked"
    print(json.dumps({"status": status, "blocker_count": len(blockers), "blockers": blockers}, sort_keys=True))
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
