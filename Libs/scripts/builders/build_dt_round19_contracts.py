#!/usr/bin/env python3
"""Freeze Round-19 scientific and training-readiness contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from jsonschema import Draft202012Validator
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config, write_json  # noqa: E402
from Libs.utils.data.round19_contracts import validate_round19_invariants  # noqa: E402


CONFIG_PATH = "Libs/configs/data/round19_public_e2_e4.yaml"
SCHEMA_REGISTRY_PATH = "Libs/configs/schema/schemas.yaml"
EXPECTED_BASE = "dba20fb554c18d18ba53b0f1cd713dbe9279d743"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _git_commit(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _resolve_output(output_root: Path, configured: object) -> Path:
    path = Path(str(configured))
    if path.is_absolute():
        raise ValueError(f"Round-19 output path must be relative: {path}")
    return output_root / path


def _validate_parent(round18_gate: Mapping[str, object], round18i_gate: Mapping[str, object], entry: Mapping[str, object]) -> None:
    blockers: list[str] = []
    if round18_gate.get("status") != "literature_held_out_sequential_replay_resolved":
        blockers.append("round18_gate")
    if round18i_gate.get("status") != "literature_replay_mainline_integrated":
        blockers.append("round18i_gate")
    if entry.get("status") != "ready_for_real_e2_e4_evidence_expansion_and_training_readiness":
        blockers.append("round19_entry")
    if entry.get("main_commit") != EXPECTED_BASE:
        blockers.append("entry_main_commit")
    if dict(entry.get("training", {})).get("authorized") is not False:
        blockers.append("training_authority")
    if dict(entry.get("evidence_fabric", {})).get("sealed") is not False:
        blockers.append("evidence_fabric")
    if entry.get("frozen_gap_count") != 20:
        blockers.append("frozen_gap_count")
    if dict(entry.get("dt_core", {})).get("status") != "not_evaluated":
        blockers.append("dt_core")
    if blockers:
        raise ValueError(f"Round-19 parent validation blocked: {sorted(blockers)}")


def build_contracts(
    *,
    repository_root: Path,
    output_root: Path,
    config_path: Path,
    round18_gate_path: Path,
    round18i_gate_path: Path,
    entry_gate_path: Path,
) -> dict[str, object]:
    config = load_config(config_path)
    section = dict(config["round19_public_e2_e4"])
    if section.get("status") != "contract_frozen_before_discovery":
        raise ValueError("Round-19 contract status is not frozen before discovery")
    if section.get("base_commit") != EXPECTED_BASE or _git_commit(repository_root) != EXPECTED_BASE:
        raise ValueError("Round-19 contracts must be frozen from the exact accepted base")

    round18_gate = _load_json(round18_gate_path)
    round18i_gate = _load_json(round18i_gate_path)
    entry_gate = _load_json(entry_gate_path)
    _validate_parent(round18_gate, round18i_gate, entry_gate)

    invariant_blockers = validate_round19_invariants(dict(section["invariants"]))
    if invariant_blockers:
        raise ValueError(invariant_blockers)

    registry_payload = yaml.safe_load(
        (repository_root / SCHEMA_REGISTRY_PATH).read_text(encoding="utf-8")
    )
    registry = registry_payload["schemas"]["round19_public_e2_e4_schema_registry"]
    scientific = dict(section["scientific_contract"])
    readiness = dict(section["training_readiness_contract"])
    for key, payload in (("scientific_contract", scientific), ("training_readiness_contract", readiness)):
        entry = dict(registry[key])
        schema = _load_json(repository_root / str(entry["path"]))
        Draft202012Validator.check_schema(schema)
        errors = sorted(Draft202012Validator(schema).iter_errors(payload), key=lambda error: list(error.path))
        if errors:
            raise ValueError([error.message for error in errors])

    outputs = dict(section["output_paths"])
    scientific_path = _resolve_output(output_root, outputs["scientific_contract"])
    readiness_path = _resolve_output(output_root, outputs["training_readiness_contract"])
    write_json(scientific_path, scientific)
    write_json(readiness_path, readiness)

    audit = {
        "schema_version": "neuronsifter-dt-round19-contract-freeze-audit-v1",
        "status": "frozen_before_discovery",
        "base_commit": EXPECTED_BASE,
        "config_path": str(config_path),
        "config_hash": sha256_file(config_path),
        "scientific_contract_path": str(scientific_path),
        "scientific_contract_hash": sha256_file(scientific_path),
        "training_readiness_contract_path": str(readiness_path),
        "training_readiness_contract_hash": sha256_file(readiness_path),
        "parent_hashes": {
            "round18_gate": sha256_file(round18_gate_path),
            "round18i_gate": sha256_file(round18i_gate_path),
            "round19_entry_gate": sha256_file(entry_gate_path),
        },
        "thresholds_mutable_after_source_observation": False,
        "discovery_started": False,
        "training_authorized": False,
        "training_started": False,
        "evidence_fabric_sealed": False,
        "dt_core_status": "not_evaluated",
    }
    write_json(output_root / "Output/audits/dt_round19_contract_freeze_audit.json", audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path(CONFIG_PATH))
    parser.add_argument("--round18-gate", type=Path, required=True)
    parser.add_argument("--round18i-gate", type=Path, required=True)
    parser.add_argument("--round19-entry-gate", type=Path, required=True)
    args = parser.parse_args()
    repository = args.repository_root.resolve()
    config_path = args.config if args.config.is_absolute() else repository / args.config
    audit = build_contracts(
        repository_root=repository,
        output_root=args.output_root.resolve(),
        config_path=config_path.resolve(),
        round18_gate_path=args.round18_gate.resolve(),
        round18i_gate_path=args.round18i_gate.resolve(),
        entry_gate_path=args.round19_entry_gate.resolve(),
    )
    print(json.dumps({"status": audit["status"], "base_commit": audit["base_commit"], "discovery_started": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
