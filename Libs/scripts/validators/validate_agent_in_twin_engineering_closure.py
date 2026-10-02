"""Validate the authoritative development Agent-in-Twin closure artifact."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any, Mapping

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.models.modules.agent_in_twin_engineering_closure import (
    ENGINEERING_CLOSURE_SCHEMA_VERSION,
    ENGINEERING_STATUS,
    FORMAL_EVIDENCE_GATE,
    SCENARIOS,
    _digest,
    sha256_file,
)
from Libs.scripts.validators.validate_agent_in_digital_twin_development import validate as validate_contract
from Libs.utils.data.raw_data_registry import load_config, resolve_repo_path


DEFAULT_CONFIG = REPO_ROOT / "Libs/configs/config_stack.yaml"
DEFAULT_SCHEMA = REPO_ROOT / "Libs/configs/schema/agent_in_twin_engineering_closure.schema.json"

_FORBIDDEN_KEYS = frozenset(
    {"chain_of_thought", "hidden_reasoning", "raw_reasoning", "scratchpad", "private_reasoning"}
)


def _find_forbidden(value: Any, path: str = "artifact") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in _FORBIDDEN_KEYS:
                found.append(f"{path}.{key}")
            found.extend(_find_forbidden(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_find_forbidden(child, f"{path}[{index}]"))
    return found


def _blockers(payload: Mapping[str, Any], config: Mapping[str, Any]) -> list[str]:
    blockers: list[str] = []
    contract = config["agent_in_digital_twin"]
    closure = contract["engineering_closure"]
    if closure.get("authoritative_status") != ENGINEERING_STATUS:
        blockers.append("config_authoritative_status_mismatch")
    if payload.get("schema_version") != ENGINEERING_CLOSURE_SCHEMA_VERSION:
        blockers.append("schema_version_mismatch")
    if payload.get("status") != ENGINEERING_STATUS:
        blockers.append("engineering_status_not_ready")
    for key, expected in (
        ("development_only", True),
        ("publication_claim_eligible", False),
        ("protected_holdout_access", 0),
        ("formal_evidence_gate", FORMAL_EVIDENCE_GATE),
        ("formal_real_world_action", "not_qualified"),
        ("formal_agent_in_twin", "not_qualified"),
        ("formal_coevolution", "not_qualified"),
    ):
        if payload.get(key) != expected:
            blockers.append(f"boundary_{key}")
    raw_scenarios = payload.get("scenarios", {})
    scenarios = raw_scenarios if isinstance(raw_scenarios, Mapping) else {}
    required = tuple(closure["required_scenarios"])
    if tuple(required) != SCENARIOS:
        blockers.append("config_scenario_contract_mismatch")
    if not isinstance(scenarios, Mapping) or len(scenarios) != len(required):
        blockers.append("scenario_count_mismatch")
    for name in required:
        if not isinstance(scenarios.get(name), Mapping) or scenarios[name].get("passed") is not True:
            blockers.append(f"scenario_{name}")
    raw_checks = payload.get("checks", {})
    checks = raw_checks if isinstance(raw_checks, Mapping) else {}
    required_checks = (
        "all_scenarios_passed",
        "scientific_state_identity_bound",
        "posterior_changes_trigger_reranking",
        "typed_tool_failures_visible",
        "memory_cannot_modify_truth",
        "unsupported_actions_abstain",
        "bidirectional_coupling_observed",
    )
    for name in required_checks:
        if checks.get(name) is not True:
            blockers.append(f"check_{name}")
    normal = scenarios.get("normal_multi_round_loop", {})
    if normal.get("initial_posterior_uid") == normal.get("post_observation_posterior_uid"):
        blockers.append("normal_loop_posterior_did_not_change")
    if normal.get("initial_posterior_digest") == normal.get("post_observation_posterior_digest"):
        blockers.append("normal_loop_digest_did_not_change")
    if normal.get("selected_action_uid_pre") == normal.get("selected_action_uid_post"):
        blockers.append("normal_loop_rank_did_not_change")
    if normal.get("ranking_posterior_uid_pre") != normal.get("initial_posterior_uid"):
        blockers.append("pre_ranking_posterior_binding")
    if normal.get("ranking_posterior_uid_post") != normal.get("post_observation_posterior_uid"):
        blockers.append("post_ranking_posterior_binding")
    if normal.get("trace_invariant") is not True:
        blockers.append("normal_loop_trace_invariant")
    if normal.get("exact_restore") is not True:
        blockers.append("normal_loop_exact_restore")
    checkpoint = payload.get("scenarios", {}).get("checkpoint_restart_replay", {})
    if (
        checkpoint.get("passed") is not True
        or checkpoint.get("exact_restore") is not True
        or checkpoint.get("memory_restore") is not True
    ):
        blockers.append("checkpoint_replay_scenario")
    raw_coevolution = scenarios.get("bidirectional_coevolution", {})
    coevolution = raw_coevolution if isinstance(raw_coevolution, Mapping) else {}
    raw_observed = coevolution.get("development_coevolution_observed_by_variant", {})
    observed = raw_observed if isinstance(raw_observed, Mapping) else {}
    if observed.get("adaptive_twin_adaptive_agent") is not True:
        blockers.append("joint_coevolution_not_observed")
    for control in (
        "frozen_twin_frozen_agent",
        "frozen_twin_adaptive_agent",
        "adaptive_twin_frozen_agent",
    ):
        if observed.get(control) is not False:
            blockers.append(f"control_{control}_coupling_label")
    return blockers


def validate(
    config_path: str | Path = DEFAULT_CONFIG,
    schema_path: str | Path = DEFAULT_SCHEMA,
    artifact_path: str | Path | None = None,
) -> dict[str, Any]:
    contract_report = validate_contract(config_path)
    config = load_config(config_path)
    closure = config["agent_in_digital_twin"]["engineering_closure"]
    resolved_artifact = resolve_repo_path(artifact_path or closure["audit_path"])
    if not resolved_artifact.exists():
        return {"status": "failed", "blockers": ["missing_engineering_closure_artifact"], "artifact_path": str(resolved_artifact)}
    payload = json.loads(resolved_artifact.read_text(encoding="utf-8"))
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors(payload), key=lambda error: list(error.path))
    blockers = [f"schema:{error.message}" for error in errors]
    if not isinstance(payload, Mapping):
        blockers.append("artifact_not_mapping")
        return {"status": "failed", "blockers": blockers, "artifact_path": str(resolved_artifact)}
    expected_digest_payload = copy.deepcopy(dict(payload))
    actual_digest = str(expected_digest_payload.pop("artifact_digest", ""))
    if actual_digest != _digest(expected_digest_payload):
        blockers.append("artifact_digest_mismatch")
    checkpoint_path = Path(str(payload.get("checkpoint_path", "")))
    if not checkpoint_path.is_absolute():
        checkpoint_path = resolve_repo_path(checkpoint_path)
    expected_checkpoint_hash = payload.get("checkpoint_sha256")
    if not checkpoint_path.exists():
        blockers.append("missing_checkpoint_artifact")
    elif not isinstance(expected_checkpoint_hash, str) or sha256_file(checkpoint_path) != expected_checkpoint_hash:
        blockers.append("checkpoint_hash_mismatch")
    blockers.extend(_blockers(payload, config))
    forbidden = _find_forbidden(payload)
    if forbidden:
        blockers.append("hidden_reasoning_fields_present")
    return {
        "status": "passed" if not blockers else "failed",
        "blockers": blockers,
        "artifact_path": str(resolved_artifact),
        "engineering_status": payload.get("status"),
        "contract_status": contract_report["engineering_closure_status"],
        "development_only": payload.get("development_only"),
        "publication_claim_eligible": payload.get("publication_claim_eligible"),
        "protected_holdout_access": payload.get("protected_holdout_access"),
        "scenario_count": len(payload.get("scenarios", {})) if isinstance(payload.get("scenarios"), Mapping) else 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA))
    parser.add_argument("--artifact", default=None)
    args = parser.parse_args()
    report = validate(args.config, args.schema, args.artifact)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
