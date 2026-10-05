"""Freeze the therapeutic evidence admission contract before acquisition."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from jsonschema import Draft202012Validator
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"
CONTRACT_CONFIG_PATH = REPO_ROOT / "Libs/configs/data/therapeutic_decision_evidence_contract.yaml"
SCHEMA_PATH = REPO_ROOT / "Libs/configs/schema/therapeutic_decision_evidence_contract.schema.json"
BUILDER_PATH = REPO_ROOT / "Libs/scripts/builders/build_therapeutic_decision_evidence_contract.py"
OUTPUT_PATH = REPO_ROOT / "Output/audits/therapeutic_decision_evidence_contract.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    return {"path": path.relative_to(REPO_ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_contract() -> dict[str, Any]:
    payload = yaml.safe_load(CONTRACT_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    contract = payload.get("therapeutic_decision_evidence_contract")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    if not isinstance(contract, dict) or not isinstance(schema, dict):
        raise TypeError("Therapeutic evidence contract and schema must be mappings")
    errors = sorted(Draft202012Validator(schema).iter_errors(contract), key=lambda error: list(error.absolute_path))
    if errors:
        detail = "; ".join(f"{list(error.absolute_path)}: {error.message}" for error in errors[:8])
        raise ValueError(f"Therapeutic evidence contract schema validation failed: {detail}")
    return contract


def _load_gate(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Gate artifact must be an object: {path}")
    return payload


def build(*, write_output: bool = True) -> dict[str, Any]:
    contract = _load_contract()
    gate_paths = {name: REPO_ROOT / path for name, path in contract["current_gate_bindings"].items()}
    gate_artifacts = {name: _artifact(path) for name, path in gate_paths.items()}
    gates = {name: _load_gate(path) for name, path in gate_paths.items()}
    targeted = gates["targeted_therapeutic_domain"]
    real = gates["real_intervention_domain"]
    split = gates["targeted_benchmark_split_audit"]
    if contract["public_boundary"]["model_predictions_inspected"] is not False:
        raise ValueError("Contract must be outcome-blind and model-prediction blind")
    if targeted.get("qualified_decision_candidate_set_count") != 0:
        raise ValueError("Current contract baseline unexpectedly contains qualified candidate sets")
    if targeted.get("development_splits_frozen") is not False or split.get("split_files_created") is not False:
        raise ValueError("A benchmark split exists before candidate qualification")
    if real.get("protected_holdout_access") != 0 or targeted.get("protected_holdout_access") != 0:
        raise ValueError("Protected holdout access is nonzero")
    contract_digest = _canonical_hash(contract)
    payload: dict[str, Any] = {
        "schema_version": "neuronsifter-therapeutic-decision-evidence-contract-audit-v2",
        "contract_version": contract["contract_version"],
        "status": "frozen_contract_current_domain_blocked",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract_digest": contract_digest,
        "contract_binding": _artifact(CONTRACT_CONFIG_PATH),
        "schema_binding": _artifact(SCHEMA_PATH),
        "implementation_binding": _artifact(BUILDER_PATH),
        "gate_bindings": gate_artifacts,
        "observed_gate": {
            "current_gate": targeted.get("status"),
            "real_intervention_decision_sensitive_sets": real.get("qualified_decision_candidate_set_count"),
            "targeted_therapeutic_qualified_sets": targeted.get("qualified_decision_candidate_set_count"),
            "benchmark_split_created": split.get("split_files_created"),
        },
        "admission_contract_frozen": True,
        "objective_selection_outcome_blind": contract["quantitative_outcome"]["objective_selection_outcome_blind"],
        "protected_holdout_access": contract["public_boundary"]["protected_holdout_access"],
        "downstream_authorization": contract["downstream_authorization"],
        "historical_round21_authority_recreated": False,
        "old_round22_tier_c_replay_used": False,
        "evidence_role": contract["public_boundary"]["evidence_role"],
        "publication_claim_eligible": contract["public_boundary"]["publication_claim_eligible"],
        "next_route": "targeted_multi_arm_matched_control_evidence_expansion",
    }
    if OUTPUT_PATH.is_file():
        existing = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        if existing.get("contract_digest") != contract_digest or existing.get("gate_bindings") != gate_artifacts:
            raise FileExistsError("Therapeutic evidence contract audit exists with a different binding")
        return existing
    if write_output:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(write_output=not args.no_write), ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
