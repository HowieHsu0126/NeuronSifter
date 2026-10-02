"""Freeze the public-only retrospective therapeutic evidence contract."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from jsonschema import Draft202012Validator
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"
CONTRACT_CONFIG_PATH = REPO_ROOT / "Libs/configs/data/retrospective_public_therapeutic_evidence.yaml"
SCHEMA_PATH = REPO_ROOT / "Libs/configs/schema/retrospective_public_therapeutic_evidence.schema.json"
LEGACY_CONTRACT_PATH = REPO_ROOT / "Libs/configs/data/therapeutic_decision_evidence_contract.yaml"
BUILDER_PATH = REPO_ROOT / "Libs/scripts/builders/build_retrospective_public_evidence_contract.py"
OUTPUT_PATH = REPO_ROOT / "Output/audits/retrospective_public_evidence_contract.json"
STATISTICAL_POLICY_PATH = REPO_ROOT / "Output/audits/retrospective_public_statistical_policy.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def canonical_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_contract() -> dict[str, Any]:
    payload = yaml.safe_load(CONTRACT_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    contract = payload.get("retrospective_public_therapeutic_evidence")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    if not isinstance(contract, dict) or not isinstance(schema, dict):
        raise TypeError("Retrospective public evidence contract and schema must be mappings")
    errors = sorted(Draft202012Validator(schema).iter_errors(contract), key=lambda error: list(error.absolute_path))
    if errors:
        detail = "; ".join(f"{list(error.absolute_path)}: {error.message}" for error in errors[:8])
        raise ValueError(f"Retrospective public evidence schema validation failed: {detail}")
    return contract


def build(*, write_output: bool = True) -> dict[str, Any]:
    contract = load_contract()
    if not LEGACY_CONTRACT_PATH.is_file():
        raise FileNotFoundError(LEGACY_CONTRACT_PATH)
    if contract["public_boundary"]["protected_holdout_access"] != 0:
        raise ValueError("Retrospective public evidence cannot access a protected holdout")
    if contract["admission"]["objective_selection_outcome_blind"] is not True:
        raise ValueError("Retrospective evidence objective selection must be outcome blind")
    if contract["summary_statistics"]["missing_variance_policy"] != "reject_formal_evidence":
        raise ValueError("Summary evidence without variance must be rejected")
    if contract["evidence_grades"]["D"]["formal_truth_eligible"] or contract["evidence_grades"]["M"]["formal_truth_eligible"]:
        raise ValueError("D and M evidence cannot be formal truth")

    contract_digest = canonical_hash(contract)
    policy_payload = {
        "schema_version": "neuronsifter-retrospective-public-statistical-policy-v1",
        "contract_digest": contract_digest,
        "formal_evidence_grades": contract["admission"]["formal_grades"],
        "non_formal_evidence_grades": contract["admission"]["non_formal_grades"],
        "effect_estimator": contract["summary_statistics"]["effect_estimator"],
        "variance_model": contract["summary_statistics"]["variance_model"],
        "sd_sem_conversion": contract["summary_statistics"]["sd_sem_conversion"],
        "ci_conversion": contract["summary_statistics"]["ci_conversion"],
        "missing_variance_policy": contract["summary_statistics"]["missing_variance_policy"],
        "replicate_bootstrap_resamples": contract["qualification"]["bootstrap_resamples"],
        "summary_bootstrap_family": contract["qualification"]["statistic_family"]["summary_grade"],
        "bootstrap_seed": contract["qualification"]["bootstrap_seed"],
        "outcome_blind": contract["admission"]["objective_selection_outcome_blind"],
        "tool_output_is_empirical_truth": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    payload: dict[str, Any] = {
        "schema_version": "neuronsifter-retrospective-public-evidence-contract-audit-v1",
        "contract_version": contract["contract_version"],
        "status": "frozen_public_retrospective_contract",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract_digest": contract_digest,
        "contract_binding": artifact(CONTRACT_CONFIG_PATH),
        "schema_binding": artifact(SCHEMA_PATH),
        "implementation_binding": artifact(BUILDER_PATH),
        "legacy_contract_binding": artifact(LEGACY_CONTRACT_PATH),
        "admission_contract_frozen": True,
        "formal_evidence_grades": contract["admission"]["formal_grades"],
        "non_formal_evidence_grades": contract["admission"]["non_formal_grades"],
        "summary_statistics_frozen": True,
        "objective_selection_outcome_blind": True,
        "author_prose_as_ground_truth": False,
        "simulator_or_docking_as_empirical_truth": False,
        "protected_holdout_access": 0,
        "publication_claim_eligible": False,
        "legacy_contract_modified": False,
        "next_route": "public_quantitative_evidence_registry_materialization",
        "output_bindings": {
            "statistical_policy": {"path": "Output/audits/retrospective_public_statistical_policy.json"},
        },
    }
    if OUTPUT_PATH.exists():
        existing = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        if existing.get("contract_digest") != contract_digest:
            raise FileExistsError("Retrospective public evidence audit exists with a different contract digest")
        if existing.get("implementation_binding") == artifact(BUILDER_PATH):
            return existing
    if write_output:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        STATISTICAL_POLICY_PATH.write_text(json.dumps(policy_payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(write_output=not args.no_write), ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
