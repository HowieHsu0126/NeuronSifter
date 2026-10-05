"""Validate the Round-23 vNext contract and development evidence boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config  # noqa: E402
from Libs.scripts.builders.build_round23_vnext_artifacts import (  # noqa: E402
    load_adaptation_outputs,
    load_development_diagnostics,
    load_framework_outputs,
    load_full_suite_execution,
    load_mechanism_outputs,
    load_training_outputs,
)


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_json(relative_path: str) -> dict[str, Any]:
    path = repo_path(relative_path)
    if not path.is_file():
        raise FileNotFoundError(relative_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {relative_path}")
    return payload


def validate(*, require_resolved: bool = False) -> dict[str, Any]:
    config = load_config(repo_path("Libs/configs/config_stack.yaml"))
    contract = config.get("neuronsifter_vnext")
    if not isinstance(contract, dict):
        raise ValueError("config_stack.yaml did not load neuronsifter_vnext")
    schema = json.loads(repo_path("Libs/configs/schema/neuronsifter_vnext_research_contract.schema.json").read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors({"neuronsifter_vnext": contract}), key=lambda error: list(error.path))
    if errors:
        raise ValueError("contract schema failure: " + "; ".join(error.message for error in errors))

    dual_gate = _load_json("Output/audits/round23_dual_gate_policy.json")
    if dual_gate["development"]["neuronsifter_core_development_authorized"] is not True or dual_gate["development"]["development_training_allowed"] is not True:
        raise ValueError("development gate is not open")
    if any(dual_gate["publication"].get(field) is not False for field in ("external_baseline_comparison_resolved", "headline_comparison_authorized", "protected_final_holdout_access")):
        raise ValueError("publication gate must remain closed")
    if dual_gate.get("protected_test_access_count") != 0:
        raise ValueError("protected holdout access must remain zero")
    if dual_gate.get("round20qr_v3_preserved") is not True:
        raise ValueError("paused Round20QR v3 root is not preserved")
    for attempt in (23, 24, 25, 26):
        if not repo_path(
            f"Output/audits/round20qr_v3/source_qualification_attempt_{attempt}"
        ).is_dir():
            raise ValueError(f"paused Round20QR v3 attempt {attempt} is not preserved")

    consistency = _load_json("Output/audits/round23_decision_consistency_contract.json")
    required_consistency = {
        "shared_posterior_identity": True,
        "hidden_scientific_decision_states": 0,
        "world_action_semantics_typed": True,
        "information_action_semantics_typed": True,
        "failed_tool_results_visible": True,
        "invalid_tool_result_moves_biological_state": False,
        "checkpoint_roundtrip_supported": True,
        "public_channel_identity_guarded": True,
        "active_posterior_rebinding_prohibited": True,
        "current_posterior_identity_read_only": True,
        "checkpoint_runtime_capability_rebinding_required": True,
    }
    for key, expected in required_consistency.items():
        if consistency.get(key) != expected:
            raise ValueError(f"decision consistency contract mismatch: {key}={consistency.get(key)!r}")

    lineage = _load_json("Output/audits/round23_posterior_lineage_validation.json")
    if lineage.get("status") != "passed" or lineage.get("decision_trace_invariant_passed") is not True:
        raise ValueError("posterior lineage smoke is not passed")
    if lineage.get("checkpoint_roundtrip") is not True:
        raise ValueError("posterior checkpoint roundtrip is not passed")
    if (
        lineage.get("direct_world_channel_stale_parent_rejected") is not True
        or lineage.get("direct_information_channel_stale_parent_rejected")
        is not True
        or lineage.get("controller_active_posterior_rebind_rejected") is not True
        or lineage.get("current_posterior_identity_read_only") is not True
        or lineage.get("checkpoint_runtime_capabilities_restored") is not True
    ):
        raise ValueError("posterior channel guard or runtime capability replay is not passed")

    framework = load_framework_outputs(contract)
    training, checkpoint_catalog, checkpoint_validation = load_training_outputs(
        contract
    )
    mechanism = load_mechanism_outputs()
    diagnostics = load_development_diagnostics()
    adaptation = load_adaptation_outputs()
    full_suite = load_full_suite_execution()
    for artifact in (training, mechanism, adaptation):
        if artifact.get("evidence_role") != "development_only" or artifact.get("publication_claim_eligible") is not False:
            raise ValueError("development artifact is missing its publication boundary")
    gate = _load_json("Output/audits/round23_vnext_development_gate.json")
    if gate["evidence_boundary"]["protected_final_holdout_accessed"] is not False or gate["evidence_boundary"]["publication_comparison_gate"] is not False:
        raise ValueError("final development gate opened a publication boundary")
    if gate["c2b"]["status"] != "blocked_insufficient_real_e2_e4":
        raise ValueError("C2-B status drifted")
    recomputed = {
        "framework": bool(
            framework["migration"]["single_canonical_framework"]
            and framework["migration"]["historical_claims_reconciled"]
        ),
        "architecture": bool(
            framework["consistency"]["shared_posterior_identity"]
            and framework["consistency"]["hidden_scientific_decision_states"] == 0
            and framework["lineage"]["status"] == "passed"
        ),
        "model": bool(
            training["c1_development_training_stable"]
            and training["c2s_development_training_stable"]
            and checkpoint_catalog["reproducible"]
            and checkpoint_validation["status"] == "passed"
            and diagnostics["complete"]
        ),
        "mechanism": bool(
            mechanism["four_way_coupling_test_complete"]
            and mechanism["coherence_diagnostics_complete"]
            and mechanism["development_gate_passed"]
        ),
        "adaptation": bool(
            adaptation["status_known"]
            and adaptation["rollback_validated_if_updates_run"]
        ),
        "full_suite": bool(full_suite.get("full_suite_green") is True),
    }
    expected_resolved = all(recomputed.values())
    if gate.get("framework", {}).get("canonical_vnext_contract") != recomputed[
        "framework"
    ]:
        raise ValueError("final Gate framework status is not evidence-bound")
    if gate.get("architecture", {}).get(
        "decision_consistency_invariant"
    ) != recomputed["architecture"]:
        raise ValueError("final Gate architecture status is not evidence-bound")
    if gate.get("model", {}).get("development_diagnostics_complete") != diagnostics[
        "complete"
    ]:
        raise ValueError("final Gate diagnostic status is not evidence-bound")
    if gate.get("mechanism", {}).get(
        "four_way_coupling_test_complete"
    ) != mechanism["four_way_coupling_test_complete"] or gate.get(
        "mechanism", {}
    ).get("coherence_diagnostics_complete") != mechanism[
        "coherence_diagnostics_complete"
    ]:
        raise ValueError("final Gate mechanism status is not evidence-bound")
    if gate.get("adaptation", {}).get("status_known") != adaptation[
        "status_known"
    ] or gate.get("adaptation", {}).get(
        "rollback_validated_if_updates_run"
    ) != adaptation[
        "rollback_validated_if_updates_run"
    ]:
        raise ValueError("final Gate adaptation status is not evidence-bound")
    if gate.get("validation", {}).get("full_suite_green") != recomputed[
        "full_suite"
    ]:
        raise ValueError("final Gate full-suite status is not execution-bound")
    if gate.get("evidence_boundary", {}).get("development_results_labeled") is not True:
        raise ValueError("final Gate development evidence labels are incomplete")
    expected_status = (
        "neuronsifter_vnext_core_development_resolved"
        if expected_resolved
        else "pending_validation_or_training"
    )
    if gate.get("status") != expected_status:
        raise ValueError("final Gate status does not match recomputed evidence")
    if require_resolved and gate.get("status") != "neuronsifter_vnext_core_development_resolved":
        raise ValueError("Round-23 development gate is not resolved")
    return {
        "status": "passed",
        "contract_id": contract["contract_id"],
        "development_gate_open": True,
        "publication_gate_closed": True,
        "posterior_lineage_passed": True,
        "mechanism_status": mechanism.get("status"),
        "training_status": training.get("large_training_stable"),
        "adaptation_status": adaptation.get("status"),
        "diagnostics_complete": diagnostics["complete"],
        "checkpoint_validation_status": checkpoint_validation["status"],
        "full_suite_green": full_suite.get("full_suite_green", False),
        "development_gate_status": gate.get("status"),
        "protected_test_access": 0,
        "c2b_status": gate["c2b"]["status"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-resolved", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate(require_resolved=args.require_resolved), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
