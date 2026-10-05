"""Validate the development-only Agent-in-Digital-Twin contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jsonschema import Draft202012Validator

from Libs.utils.data.raw_data_registry import load_config, resolve_repo_path

DEFAULT_CONFIG = REPO_ROOT / "Libs/configs/config_stack.yaml"
DEFAULT_SCHEMA = REPO_ROOT / "Libs/configs/schema/agent_in_digital_twin_development.schema.json"


def validate(config_path: str | Path = DEFAULT_CONFIG, schema_path: str | Path = DEFAULT_SCHEMA) -> dict[str, Any]:
    config = load_config(config_path)
    if "agent_in_digital_twin" not in config:
        raise ValueError("config stack is missing agent_in_digital_twin")
    payload = {"agent_in_digital_twin": config["agent_in_digital_twin"]}
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(payload), key=lambda error: list(error.path))
    if errors:
        raise ValueError("Agent-in-Digital-Twin schema errors: " + "; ".join(error.message for error in errors))
    contract = payload["agent_in_digital_twin"]
    boundaries = tuple(str(value) for value in contract["mcp_trust_boundaries"])
    if len(boundaries) != 6 or len(set(boundaries)) != 6:
        raise ValueError("The six MCP trust boundaries must be unique")
    fields = tuple(str(value) for value in contract["belief_invariant"]["fields"])
    if len(fields) != 5 or len(set(fields)) != 5:
        raise ValueError("The five posterior invariant fields must be unique")
    closure = contract["engineering_closure"]
    if len(closure["required_scenarios"]) != 8 or len(set(closure["required_scenarios"])) != 8:
        raise ValueError("Engineering closure must freeze eight unique replay scenarios")
    return {
        "status": "passed",
        "mode": contract["mode"],
        "formal_evidence_gate": contract["formal_evidence_gate"],
        "publication_claim_eligible": contract["publication_claim_eligible"],
        "protected_holdout_access": contract["protected_holdout_access"],
        "mcp_trust_boundaries": list(boundaries),
        "belief_invariant_fields": list(fields),
        "coevolution_variants": list(contract["coevolution"]["variants"]),
        "engineering_closure_status": closure["authoritative_status"],
        "engineering_closure_scenarios": list(closure["required_scenarios"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA))
    args = parser.parse_args()
    print(json.dumps(validate(args.config, args.schema), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
