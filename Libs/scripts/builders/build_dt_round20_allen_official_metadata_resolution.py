#!/usr/bin/env python3
"""Build Round-20 Allen official metadata rights/release resolution artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.round20_allen_official_metadata_resolution import CONFIG_KEY, CONFIG_RELATIVE, emit_artifacts


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
    args = parser.parse_args()
    repository = args.repository_root.resolve()
    contract = load_contract(repository, args.config)
    audit = emit_artifacts(contract, repository=repository)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "gate": audit["gate"],
                "family_count": audit["family_count"],
                "blocker_count": audit["blocker_count"],
                "output": str(repository / str(contract["output_artifacts"]["resolution_audit"])),
            },
            sort_keys=True,
        )
    )
    return 0 if audit["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
