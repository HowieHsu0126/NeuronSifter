"""Run the development-only Agent-in-Digital-Twin engineering closure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.models.modules.agent_in_twin_engineering_closure import run_engineering_closure
from Libs.scripts.validators.validate_agent_in_digital_twin_development import validate as validate_contract
from Libs.utils.data.raw_data_registry import load_config, resolve_repo_path


DEFAULT_CONFIG = REPO_ROOT / "Libs/configs/config_stack.yaml"


def run(*, config_path: str | Path = DEFAULT_CONFIG, write: bool = True) -> dict[str, Any]:
    contract = validate_contract(config_path)
    config = load_config(config_path)
    closure = config["agent_in_digital_twin"]["engineering_closure"]
    output_root = resolve_repo_path(closure["output_root"])
    artifact = run_engineering_closure(output_root if write else None)
    if write:
        audit_path = resolve_repo_path(closure["audit_path"])
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    artifact = run(config_path=args.config, write=not args.no_write)
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "development_only": artifact["development_only"],
                "publication_claim_eligible": artifact["publication_claim_eligible"],
                "protected_holdout_access": artifact["protected_holdout_access"],
                "scenario_count": len(artifact["scenarios"]),
            },
            sort_keys=True,
        )
    )
    return 0 if artifact["status"] == "ENGINEERING_READY_AWAITING_EXTERNAL_THERAPEUTIC_EVIDENCE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
