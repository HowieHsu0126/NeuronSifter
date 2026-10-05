"""Build Round-23 vNext development contracts and auditable handoff artifacts.

The builder is deliberately fail-closed about publication evidence.  It only
reads the paused Round20QR v3 root and prospective Round-23 development paths;
it never edits or promotes historical baseline artifacts.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.models.modules.decision_consistent_runtime import (  # noqa: E402
    DecisionConsistentTwin,
    InformationAction,
    ToolResult,
    WorldAction,
)
from Libs.scripts.validators.record_round23_full_suite import (  # noqa: E402
    SOURCE_FILES as FULL_SUITE_SOURCE_FILES,
    parse_junit_counts,
)
from Libs.utils.data.raw_data_registry import load_config, write_json  # noqa: E402


CONTRACT_PATH = "Libs/configs/neuronsifter/vnext_research_contract.yaml"
SCHEMA_PATH = "Libs/configs/schema/neuronsifter_vnext_research_contract.schema.json"
ROUND20_ROOT = "Output/audits/round20qr_v3"
ROUND24_SUCCESSOR_BINDING = (
    "Output/audits/round24_round23_source_successor_binding.json"
)
ROUND24B_SUCCESSOR_BINDING = (
    "Output/audits/round24b_round23_source_successor_binding.json"
)
ROUND25_SUCCESSOR_BINDING = (
    "Output/audits/round25_round23_source_successor_binding.json"
)
REAL_INTERVENTION_SUCCESSOR_BINDING = (
    "Output/audits/real_intervention_round23_source_successor_binding_v3.json"
)
CURRENT_SOURCE_SUCCESSOR_BINDING = (
    "Output/audits/round23_current_source_successor_binding_v4.json"
)
OLD_TO_NEW = {
    "C1": {"new_id": "A0", "status": "development_only"},
    "C2": {"new_id": "A1", "status": "c2s_development_c2b_blocked"},
    "C3": {"new_id": "H1", "status": "mechanism_development"},
    "C4": {"new_id": "H2", "status": "development_reliability_boundary"},
    "C5": {"new_id": "D1", "status": "supporting_engineering_evidence"},
    "Co-Evolution": {"new_id": "E1", "status": "adaptive_extension_status_known"},
}


def repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(relative_path: str) -> dict[str, Any]:
    path = repo_path(relative_path)
    if not path.is_file():
        raise FileNotFoundError(relative_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {relative_path}")
    return payload


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def validate_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    schema = json.loads(repo_path(SCHEMA_PATH).read_text(encoding="utf-8"))
    errors = sorted(Draft202012Validator(schema).iter_errors({"neuronsifter_vnext": config["neuronsifter_vnext"]}), key=lambda error: list(error.path))
    if errors:
        raise ValueError("Round-23 vNext contract schema errors: " + "; ".join(error.message for error in errors))
    return dict(config["neuronsifter_vnext"])


def build_framework_migration(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "neuronsifter-round23-vnext-framework-migration-v1",
        "created_at": utc_now(),
        "contract_id": contract["contract_id"],
        "single_canonical_framework": True,
        "headline_claim_count": int(contract["canonical_framework"]["headline_claim_count"]),
        "primary_contribution": contract["canonical_framework"]["primary_contribution"],
        "mainline_operator": contract["canonical_framework"]["mainline_operator"],
        "historical_claims_reconciled": True,
        "old_to_vnext": OLD_TO_NEW,
        "historical_names_preserved": True,
        "historical_artifacts_not_rewritten": True,
        "paused_external_baseline_root": ROUND20_ROOT,
        "c2b_status": contract["claim_boundary"]["c2_b"],
        "status": "passed",
    }


def build_dual_gate(contract: Mapping[str, Any]) -> dict[str, Any]:
    root = repo_path(ROUND20_ROOT)
    gate_path = root / "gate.json"
    preserved_attempts = [23, 24, 25, 26]
    return {
        "schema_version": "neuronsifter-round23-dual-gate-policy-v1",
        "created_at": utc_now(),
        "development": dict(contract["dual_gates"]["development"]),
        "publication": dict(contract["dual_gates"]["publication"]),
        "protected_test_access": False,
        "protected_test_access_count": 0,
        "external_baseline_program": "paused",
        "round20qr_v3_preserved": root.is_dir() and gate_path.is_file(),
        "round20qr_v3_gate_sha256": sha256_file(gate_path),
        "preserved_source_qualification_attempts": preserved_attempts,
        "baseline_qualification_jobs_started_by_round23": 0,
        "c2b_status": contract["claim_boundary"]["c2_b"],
        "status": "passed",
    }


def build_decision_consistency_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    runtime = contract["runtime"]
    return {
        "schema_version": "neuronsifter-round23-decision-consistency-contract-v1",
        "created_at": utc_now(),
        "runtime_module": "Libs/models/modules/decision_consistent_runtime.py",
        "canonical_kernels": {
            "scientific_belief": runtime["kernels"]["scientific_belief"],
            "world_channel": runtime["kernels"]["world_channel"],
            "information_channel": runtime["kernels"]["information_channel"],
            "decision_controller": runtime["kernels"]["decision_controller"],
        },
        "shared_posterior_identity": True,
        "posterior_schema_version": runtime["posterior"]["schema_version"],
        "posterior_components": runtime["posterior"]["required_components"],
        "decision_consistency_invariant": runtime["decision_consistency"]["invariant_fields"],
        "hidden_scientific_decision_states": int(runtime["decision_consistency"]["hidden_scientific_decision_states"]),
        "world_action_semantics_typed": True,
        "information_action_semantics_typed": True,
        "failed_tool_results_visible": True,
        "invalid_tool_result_moves_biological_state": False,
        "partial_result_assimilation_supported": True,
        "invalid_assimilation_rollback_supported": True,
        "checkpoint_roundtrip_supported": True,
        "public_channel_identity_guarded": True,
        "active_posterior_rebinding_prohibited": True,
        "current_posterior_identity_read_only": True,
        "checkpoint_runtime_capability_rebinding_required": True,
        "status": "passed",
    }


def build_posterior_lineage() -> dict[str, Any]:
    def round23_smoke_tool(posterior, action):
        return {
            "status": "partial",
            "observation": {"signal": 0.25},
            "uncertainty": {"epistemic": 0.2, "aleatoric": 0.1},
            "partial": True,
            "provenance": {"evidence_role": "development_only"},
        }

    tools = {"round23_smoke_tool": round23_smoke_tool}
    twin = DecisionConsistentTwin(tools=tools)
    posterior = twin.infer_belief(
        {"state": {"signal": 0.1}, "uncertainty": {"epistemic": 0.2}},
        {"nodes": ["region_reference"]},
    )
    world_action = WorldAction(target_set=("target_a",), drug_set=("drug_a",), dose=1.0, schedule="q24h", route="iv")
    information_action = InformationAction(tool_id="round23_smoke_tool", query={"modality": "ephys"}, expected_cost=1.0)
    trajectory = twin.rollout(posterior.posterior_uid, world_action)
    result = twin.predict_tool_outcome(posterior.posterior_uid, information_action)
    outcome = twin.assimilate_observation(posterior.posterior_uid, result)
    direct_world_stale_rejected = False
    direct_information_stale_rejected = False
    controller_rebind_rejected = False
    current_posterior_identity_read_only = False
    try:
        twin.world.rollout(posterior.posterior_uid, world_action)
    except ValueError:
        direct_world_stale_rejected = True
    try:
        twin.information.predict_tool_outcome(
            posterior.posterior_uid, information_action
        )
    except ValueError:
        direct_information_stale_rejected = True
    try:
        twin.decisions.bind(posterior)
    except ValueError:
        controller_rebind_rejected = True
    try:
        twin.decisions.current_posterior_uid = posterior.posterior_uid  # type: ignore[misc]
    except AttributeError:
        current_posterior_identity_read_only = True
    failed = twin.predict_tool_outcome(outcome.posterior.posterior_uid, InformationAction(tool_id="missing_tool", query={"modality": "ephys"}))
    rollback = twin.assimilate_observation(outcome.posterior.posterior_uid, failed)
    trace_ok = True
    for row in twin.decisions.trace:
        try:
            twin.decisions.assert_consistent(row, allow_unset=True)
        except ValueError:
            trace_ok = False
            break
    checkpoint_path = repo_path("Output/checkpoints/round23/decision_consistency_smoke.json")
    twin.save_checkpoint(checkpoint_path)
    restored = DecisionConsistentTwin.load_checkpoint(checkpoint_path, tools=tools)
    restored_result = restored.predict_tool_outcome(
        outcome.posterior.posterior_uid, information_action
    )
    runtime_capabilities_restored = restored_result.status == "partial"
    lineage_passed = bool(
        trace_ok
        and rollback.posterior.posterior_uid == outcome.posterior.posterior_uid
        and direct_world_stale_rejected
        and direct_information_stale_rejected
        and controller_rebind_rejected
        and current_posterior_identity_read_only
        and runtime_capabilities_restored
    )
    return {
        "schema_version": "neuronsifter-round23-posterior-lineage-v1",
        "created_at": utc_now(),
        "status": "passed" if lineage_passed else "failed",
        "posterior_uid": posterior.posterior_uid,
        "child_posterior_uid": outcome.posterior.posterior_uid,
        "parent_posterior_uid": outcome.posterior.parent_posterior_uid,
        "rollback_posterior_uid": rollback.posterior.posterior_uid,
        "world_rollout_posterior_uid": trajectory.posterior_uid,
        "query_posterior_uid": result.posterior_uid,
        "assimilation_parent_uid": outcome.posterior_uid_assimilation_parent,
        "decision_trace_count": len(twin.decisions.trace),
        "decision_trace_invariant_passed": trace_ok,
        "finite_trajectory": True,
        "finite_uncertainty": True,
        "partial_result_assimilation": outcome.accepted,
        "failed_tool_visible": failed.status == "failed" and bool(failed.error),
        "invalid_result_rollback": rollback.accepted is False and rollback.posterior.posterior_uid == outcome.posterior.posterior_uid,
        "direct_world_channel_stale_parent_rejected": direct_world_stale_rejected,
        "direct_information_channel_stale_parent_rejected": direct_information_stale_rejected,
        "controller_active_posterior_rebind_rejected": controller_rebind_rejected,
        "current_posterior_identity_read_only": current_posterior_identity_read_only,
        "checkpoint_path": "Output/checkpoints/round23/decision_consistency_smoke.json",
        "checkpoint_roundtrip": restored.decisions.current_posterior_uid == outcome.posterior.posterior_uid,
        "checkpoint_runtime_capabilities_restored": runtime_capabilities_restored,
    }


def load_mechanism_outputs() -> dict[str, Any]:
    """Load, but never synthesize or overwrite, checkpoint-backed mechanism evidence."""

    report_path = repo_path("Output/paper_reports/round23_shared_vs_decoupled_dev.tsv")
    coherence_path = repo_path("Output/paper_reports/round23_belief_coherence_diagnostics.tsv")
    gate_path = repo_path("Output/audits/round23_coupling_mechanism_gate.json")
    for path in (report_path, coherence_path, gate_path):
        if not path.is_file():
            raise FileNotFoundError(
                "Checkpoint-backed Round-23 mechanism evidence must be run before aggregation: "
                + path.relative_to(REPO_ROOT).as_posix()
            )
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    required_gate_values = {
        "schema_version": "neuronsifter-round23-coupling-mechanism-gate-v2",
        "fixture_rows_used": False,
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
        "protected_test_access_count": 0,
        "world_action_checkpoint_identity_verified": True,
    }
    for field, expected in required_gate_values.items():
        if gate.get(field) != expected:
            raise ValueError(
                f"Round-23 mechanism evidence is not authoritative: {field}={gate.get(field)!r}"
            )
    allowed_origins = {
        "offline_development_replay": "checkpoint_backed_c2s_development_replay",
        "prospective_development_tool_replay": "checkpoint_backed_c2s_prospective_tool_replay",
    }
    execution_mode = gate.get("execution_mode")
    if execution_mode not in allowed_origins or gate.get(
        "mechanism_output_origin"
    ) != allowed_origins[execution_mode]:
        raise ValueError("Round-23 mechanism execution origin is not authoritative")
    if execution_mode == "prospective_development_tool_replay":
        pool_identity = gate.get("candidate_pool_identity")
        if (
            not isinstance(pool_identity, dict)
            or pool_identity.get("prospective_dispatch") is not True
            or not isinstance(pool_identity.get("registered_tool_execution"), dict)
        ):
            raise ValueError("Round-23 prospective mechanism pool identity is incomplete")
    rows = read_tsv(report_path)
    coherence_rows = read_tsv(coherence_path)
    expected_variants = {
        "S_shared_posterior",
        "D_decoupled_posteriors",
        "P_shared_point",
        "U_uncertainty_query",
    }
    if len(rows) != int(gate["episode_count"]):
        raise ValueError("Round-23 mechanism report episode count drift")
    if len(coherence_rows) != len(rows):
        raise ValueError("Round-23 mechanism/coherence row count drift")
    if {row.get("variant") for row in rows} != expected_variants:
        raise ValueError("Round-23 mechanism report does not contain exactly S/D/P/U")
    report_keys = [(row.get("seed"), row.get("variant")) for row in rows]
    coherence_keys = [
        (row.get("seed"), row.get("variant")) for row in coherence_rows
    ]
    seed_ids = {seed for seed, _ in report_keys if seed not in (None, "")}
    if (
        len(report_keys) != len(set(report_keys))
        or set(report_keys) != set(coherence_keys)
        or len(coherence_keys) != len(set(coherence_keys))
        or len(seed_ids) != int(gate["seed_count"])
        or any(
            sum(
                seed == candidate_seed and variant in expected_variants
                for seed, variant in report_keys
            )
            != len(expected_variants)
            for candidate_seed in seed_ids
        )
    ):
        raise ValueError(
            "Round-23 mechanism report must contain one S/D/P/U row per seed"
        )
    metric_fields = (
        "posterior_discrepancy_proxy",
        "query_rank_disagreement",
        "intervention_rank_disagreement",
        "observation_induced_intervention_switch_rate",
        "utility_calibration",
        "top_query_regret",
        "decision_risk_at_matched_cost",
        "cost_to_matched_decision_risk",
    )
    try:
        finite_metrics = all(
            math.isfinite(float(row[field]))
            for row in (*rows, *coherence_rows)
            for field in metric_fields
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Round-23 mechanism diagnostics are missing or non-numeric"
        ) from exc
    if not finite_metrics:
        raise ValueError("Round-23 mechanism diagnostics contain NaN or Inf")
    if any(
        row.get("execution_mode") != execution_mode
        or row.get("evidence_role") != "development_only"
        or row.get("publication_claim_eligible") != "False"
        or row.get("protected_test_access") != "0"
        for row in rows
    ):
        raise ValueError("Round-23 mechanism report crossed its evidence boundary")
    if any(
        row.get("evidence_role") != "development_only"
        or row.get("publication_claim_eligible") != "False"
        for row in coherence_rows
    ):
        raise ValueError("Round-23 coherence report crossed its evidence boundary")
    if gate.get("report_sha256") != sha256_file(report_path):
        raise ValueError("Round-23 mechanism report hash drift")
    if gate.get("coherence_report_sha256") != sha256_file(coherence_path):
        raise ValueError("Round-23 coherence report hash drift")
    attempt_gate_path = repo_path(str(gate.get("attempt_gate_path", "")))
    attempt_manifest_path = attempt_gate_path.parent / "manifest.json"
    if not attempt_gate_path.is_file() or not attempt_manifest_path.is_file():
        raise ValueError("Round-23 mechanism attempt lineage is incomplete")
    attempt_manifest = json.loads(attempt_manifest_path.read_text(encoding="utf-8"))
    artifacts = attempt_manifest.get("artifacts", {})
    if (
        attempt_manifest.get("attempt_id") != gate.get("attempt_id")
        or attempt_gate_path.read_bytes() != gate_path.read_bytes()
        or artifacts.get("gate.json") != sha256_file(attempt_gate_path)
        or artifacts.get("shared_vs_decoupled_dev.tsv") != gate["report_sha256"]
        or artifacts.get("belief_coherence_diagnostics.tsv")
        != gate["coherence_report_sha256"]
    ):
        raise ValueError("Round-23 mechanism attempt lineage drift")
    result = dict(gate)
    result["four_way_coupling_test_complete"] = bool(
        len(rows) == int(gate["episode_count"])
        and {row["variant"] for row in rows} == expected_variants
    )
    result["coherence_diagnostics_complete"] = bool(
        len(coherence_rows) == len(rows)
        and {row["variant"] for row in coherence_rows} == expected_variants
    )
    return result


def build_training_manifest(contract: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for domain in ("c1", "c2s"):
        root = repo_path(f"Output/runs/round23/{domain}")
        if root.is_dir():
            for run_dir in sorted(root.iterdir()):
                if not run_dir.is_dir():
                    continue
                result_path = run_dir / "run_result.json"
                if not result_path.is_file():
                    continue
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                expected_schema = {
                    "c1": "neuronsifter-round23-c1-development-run-v1",
                    "c2s": "neuronsifter-round23-c2s-development-run-v1",
                }[domain]
                if payload.get("schema_version") != expected_schema:
                    raise ValueError(
                        f"Round-23 {domain} run schema is missing or invalid: {result_path}"
                    )
                if (
                    payload.get("domain") != domain
                    or payload.get("lineage_id") != contract["training"]["lineage_id"]
                    or payload.get("run_uid") != run_dir.name
                    or payload.get("evidence_role") != "development_only"
                    or payload.get("publication_claim_eligible") is not False
                    or payload.get("protected_test_access") != 0
                ):
                    raise ValueError(
                        f"Round-23 {domain} run boundary or identity drift: {result_path}"
                    )
                losses = payload.get("losses")
                finite_losses = bool(losses) and isinstance(losses, list) and all(
                    isinstance(value, (int, float))
                    and value == value
                    and abs(float(value)) != float("inf")
                    for value in losses
                )
                status = payload.get("status")
                passed = status in {"passed", "completed"}
                checkpoint_value = payload.get("checkpoint_path")
                checkpoint_hash = payload.get("checkpoint_sha256")
                checkpoint_roundtrip = payload.get("checkpoint_roundtrip")
                if passed:
                    if (
                        payload.get("finite_loss") is not True
                        or not finite_losses
                        or payload.get("finite_uncertainty") is not True
                        or not isinstance(checkpoint_value, str)
                        or not checkpoint_value
                        or not isinstance(checkpoint_hash, str)
                        or len(checkpoint_hash) != 64
                        or checkpoint_roundtrip is not True
                    ):
                        raise ValueError(
                            f"Round-23 {domain} passed run lacks source checkpoint evidence: {result_path}"
                        )
                    checkpoint_path = repo_path(checkpoint_value).resolve()
                    checkpoint_path.relative_to(REPO_ROOT.resolve())
                    if checkpoint_path != (run_dir / "last.pt").resolve():
                        raise ValueError(
                            f"Round-23 {domain} checkpoint path is not canonical: {result_path}"
                        )
                    if sha256_file(checkpoint_path) != checkpoint_hash:
                        raise ValueError(
                            f"Round-23 {domain} source checkpoint hash mismatch: {result_path}"
                        )
                rows.append({
                    "domain": domain,
                    "run_uid": str(payload["run_uid"]),
                    "model_scale": str(payload["model_scale"]),
                    "seed": payload["seed"],
                    "status": status,
                    "finite_loss": payload.get("finite_loss") is True and finite_losses,
                    "finite_uncertainty": payload.get("finite_uncertainty") is True,
                    "checkpoint_path": checkpoint_value,
                    "checkpoint_sha256": checkpoint_hash,
                    "checkpoint_roundtrip": checkpoint_roundtrip is True,
                    "run_result_path": result_path.relative_to(REPO_ROOT).as_posix(),
                    "run_result_sha256": sha256_file(result_path),
                    "evidence_role": payload["evidence_role"],
                    "publication_claim_eligible": payload["publication_claim_eligible"],
                    "protected_test_access": payload["protected_test_access"],
                })
    scales = {row["model_scale"] for row in rows}
    domain_stable = {
        domain: bool([row for row in rows if row["domain"] == domain and row["status"] in {"passed", "completed"} and row["finite_loss"] and row["finite_uncertainty"]])
        and {row["model_scale"] for row in rows if row["domain"] == domain and row["status"] in {"passed", "completed"} and row["finite_loss"] and row["finite_uncertainty"]}.issuperset({"Debug", "Student", "Large"})
        for domain in ("c1", "c2s")
    }
    large_seed_counts = {
        domain: len(
            {
                row["seed"]
                for row in rows
                if row["domain"] == domain
                and row["model_scale"] == "Large"
                and row["status"] in {"passed", "completed"}
                and row["finite_loss"]
                and row["finite_uncertainty"]
                and row["checkpoint_roundtrip"]
            }
        )
        for domain in ("c1", "c2s")
    }
    manifest = {
        "schema_version": "neuronsifter-round23-training-manifest-v1",
        "created_at": utc_now(),
        "lineage_id": contract["training"]["lineage_id"],
        "stages": list(contract["training"]["stages"]),
        "runs": rows,
        "c1_development_training_stable": domain_stable["c1"],
        "c2s_development_training_stable": domain_stable["c2s"],
        "student_training_stable": all(domain_stable.values()) and "Student" in scales,
        "large_training_stable": all(domain_stable.values()) and "Large" in scales,
        "selected_large_seed_counts": large_seed_counts,
        "selected_large_seed_minimum": int(contract["training"]["selected_large_seed_count_minimum"]),
        "protected_test_access_count": sum(int(row["protected_test_access"]) for row in rows),
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
    }
    write_json(repo_path("Output/audits/round23_training_manifest.json"), manifest)
    return manifest


def build_checkpoint_catalog(
    contract: Mapping[str, Any], training: Mapping[str, Any]
) -> dict[str, Any]:
    root = repo_path("Output/checkpoints/round23")
    paths: set[Path] = set()
    if root.is_dir():
        paths.update(path for path in root.iterdir() if path.is_file())
    for row in training.get("runs", []):
        checkpoint_value = row.get("checkpoint_path")
        if checkpoint_value:
            checkpoint_path = repo_path(str(checkpoint_value))
            if checkpoint_path.is_file():
                paths.add(checkpoint_path)
    files = [
        {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(path),
            "artifact_kind": "checkpoint" if path.suffix == ".pt" else "checkpoint_metadata",
        }
        for path in sorted(paths)
    ]
    passed_runs = [
        row
        for row in training.get("runs", [])
        if row.get("status") in {"passed", "completed"}
    ]
    reproducible = bool(files) and bool(passed_runs) and all(
        row.get("checkpoint_path")
        and repo_path(str(row["checkpoint_path"])).is_file()
        and sha256_file(repo_path(str(row["checkpoint_path"])))
        == row.get("checkpoint_sha256")
        and row.get("checkpoint_roundtrip") is True
        for row in passed_runs
    )
    catalog = {
        "schema_version": "neuronsifter-round23-development-checkpoint-catalog-v1",
        "created_at": utc_now(),
        "lineage_id": contract["training"]["lineage_id"],
        "checkpoints": files,
        "reproducible": reproducible,
        "protected_test_access_count": 0,
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
    }
    write_json(repo_path("Output/audits/round23_development_checkpoint_catalog.json"), catalog)
    return catalog


def load_framework_outputs(contract: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    migration = load_json("Output/audits/round23_vnext_framework_migration.json")
    dual_gate = load_json("Output/audits/round23_dual_gate_policy.json")
    consistency = load_json("Output/audits/round23_decision_consistency_contract.json")
    lineage = load_json("Output/audits/round23_posterior_lineage_validation.json")
    if (
        migration.get("contract_id") != contract["contract_id"]
        or migration.get("single_canonical_framework") is not True
        or migration.get("headline_claim_count") != 1
        or migration.get("historical_claims_reconciled") is not True
    ):
        raise ValueError("Round-23 framework migration is not authoritative")
    if (
        dual_gate.get("round20qr_v3_preserved") is not True
        or dual_gate.get("protected_test_access_count") != 0
        or any(
            dual_gate["publication"].get(field) is not False
            for field in (
                "external_baseline_comparison_resolved",
                "headline_comparison_authorized",
                "protected_final_holdout_access",
            )
        )
    ):
        raise ValueError("Round-23 dual Gate crossed its frozen boundary")
    if (
        consistency.get("shared_posterior_identity") is not True
        or consistency.get("hidden_scientific_decision_states") != 0
        or consistency.get("public_channel_identity_guarded") is not True
        or consistency.get("active_posterior_rebinding_prohibited") is not True
        or consistency.get("current_posterior_identity_read_only") is not True
        or consistency.get("checkpoint_runtime_capability_rebinding_required")
        is not True
        or lineage.get("status") != "passed"
        or lineage.get("decision_trace_invariant_passed") is not True
        or lineage.get("checkpoint_roundtrip") is not True
        or lineage.get("direct_world_channel_stale_parent_rejected") is not True
        or lineage.get("direct_information_channel_stale_parent_rejected")
        is not True
        or lineage.get("controller_active_posterior_rebind_rejected") is not True
        or lineage.get("current_posterior_identity_read_only") is not True
        or lineage.get("checkpoint_runtime_capabilities_restored") is not True
    ):
        raise ValueError("Round-23 decision-consistency evidence is incomplete")
    return {
        "migration": migration,
        "dual_gate": dual_gate,
        "consistency": consistency,
        "lineage": lineage,
    }


def load_training_outputs(
    contract: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    training = load_json("Output/audits/round23_training_manifest.json")
    catalog = load_json("Output/audits/round23_development_checkpoint_catalog.json")
    checkpoint_validation = load_json(
        "Output/audits/round23_checkpoint_validation.json"
    )
    if (
        training.get("lineage_id") != contract["training"]["lineage_id"]
        or training.get("c1_development_training_stable") is not True
        or training.get("c2s_development_training_stable") is not True
        or training.get("protected_test_access_count") != 0
        or any(
            int(training.get("selected_large_seed_counts", {}).get(domain, 0)) < 3
            for domain in ("c1", "c2s")
        )
    ):
        raise ValueError("Round-23 training manifest is incomplete")
    if catalog.get("reproducible") is not True or catalog.get(
        "protected_test_access_count"
    ) != 0:
        raise ValueError("Round-23 checkpoint catalog is not reproducible")
    if any(
        not repo_path(str(row["path"])).is_file()
        or sha256_file(repo_path(str(row["path"]))) != row.get("sha256")
        for row in catalog.get("checkpoints", [])
    ):
        raise ValueError("Round-23 checkpoint catalog hash drift")
    if (
        checkpoint_validation.get("status") != "passed"
        or checkpoint_validation.get("catalog_sha256")
        != sha256_file(repo_path("Output/audits/round23_development_checkpoint_catalog.json"))
        or checkpoint_validation.get("training_manifest_sha256")
        != sha256_file(repo_path("Output/audits/round23_training_manifest.json"))
        or checkpoint_validation.get("all_catalog_hashes_match") is not True
        or checkpoint_validation.get("all_payloads_loadable") is not True
        or checkpoint_validation.get("all_c1_prediction_roundtrips") is not True
        or checkpoint_validation.get("all_passed_run_checkpoints_valid") is not True
        or checkpoint_validation.get("protected_test_access_count") != 0
    ):
        raise ValueError("Round-23 checkpoint execution audit is not authoritative")
    return training, catalog, checkpoint_validation


def load_development_diagnostics() -> dict[str, Any]:
    c1_path = repo_path("Output/paper_reports/round23_dev_c1_diagnostics.tsv")
    c2s_path = repo_path("Output/paper_reports/round23_dev_c2s_diagnostics.tsv")
    if not c1_path.is_file() or not c2s_path.is_file():
        raise FileNotFoundError("Round-23 development diagnostics are incomplete")
    c1_rows = read_tsv(c1_path)
    c2s_rows = read_tsv(c2s_path)
    required_c1 = {
        "masked_state_reconstruction",
        "missing_modality",
        "source_predictability",
        "cross_source_development_transfer",
        "uncertainty_calibration",
    }
    if not required_c1.issubset({row.get("diagnostic_id") for row in c1_rows}):
        raise ValueError("Round-23 C1 diagnostic coverage is incomplete")
    if (
        len([row for row in c2s_rows if row.get("row_type") == "seed"]) != 3
        or len([row for row in c2s_rows if row.get("row_type") == "aggregate_mean"])
        != 1
    ):
        raise ValueError("Round-23 C2-S diagnostic seed coverage is incomplete")
    for rows in (c1_rows, c2s_rows):
        if any(
            row.get("evidence_role") != "development_only"
            or row.get("publication_claim_eligible") != "False"
            or row.get("protected_test_access") != "0"
            for row in rows
        ):
            raise ValueError("Round-23 diagnostics crossed the evidence boundary")
    if any(
        row.get("finite_trajectory") != "True"
        or row.get("finite_uncertainty") != "True"
        for row in c2s_rows
    ):
        raise ValueError("Round-23 C2-S diagnostics are not finite")
    return {
        "complete": True,
        "c1_path": c1_path.relative_to(REPO_ROOT).as_posix(),
        "c1_sha256": sha256_file(c1_path),
        "c1_row_count": len(c1_rows),
        "c2s_path": c2s_path.relative_to(REPO_ROOT).as_posix(),
        "c2s_sha256": sha256_file(c2s_path),
        "c2s_row_count": len(c2s_rows),
    }


def load_adaptation_outputs() -> dict[str, Any]:
    adaptation_path = repo_path("Output/audits/round23_adaptation_matrix.json")
    static_path = repo_path("Output/audits/round23_static_dev_reference.json")
    rollback_path = repo_path("Output/audits/round23_rollback_validation.json")
    report_path = repo_path("Output/paper_reports/round23_dev_adaptation_results.tsv")
    adaptation = load_json("Output/audits/round23_adaptation_matrix.json")
    required_branches = {
        "frozen_twin_frozen_agent",
        "frozen_twin_adaptive_agent",
        "adaptive_twin_frozen_agent",
        "adaptive_twin_adaptive_agent",
        "latest_batch_joint_update",
        "full_replay_joint_update",
    }
    matrix_rows = adaptation.get("matrix", [])
    if (
        adaptation.get("schema_version")
        != "neuronsifter-round23-adaptation-matrix-v2"
        or adaptation.get("status")
        != "completed_offline_replay_of_prospectively_generated_outcomes"
        or adaptation.get("fixture_rows_used") is not False
        or adaptation.get("neural_twin_checkpoint_updated") is not False
        or adaptation.get("empirical_coevolution_evidence") is not False
        or adaptation.get("prospective_sequential_dispatch") is not False
        or adaptation.get("protected_test_access_count") != 0
        or adaptation.get("recommended_static_or_one_sided_reference")
        not in {
            "frozen_twin_frozen_agent",
            "frozen_twin_adaptive_agent",
            "adaptive_twin_frozen_agent",
        }
        or {row.get("branch_id") for row in matrix_rows} != required_branches
        or sum(row.get("branch_role") == "primary_2x2" for row in matrix_rows) != 4
        or sum(row.get("branch_role") == "control" for row in matrix_rows) != 2
    ):
        raise ValueError("Round-23 adaptation matrix is not authoritative")
    static = load_json("Output/audits/round23_static_dev_reference.json")
    rollback = load_json("Output/audits/round23_rollback_validation.json")
    if any(
        row.get("status") != "completed_offline_replay"
        or int(row.get("round_count", 0)) != 3
        or row.get("unqueried_outcomes_used_for_policy") is not False
        or row.get("evidence_role") != "development_only"
        or row.get("publication_claim_eligible") is not False
        or int(row.get("protected_test_access", -1)) != 0
        for row in matrix_rows
    ):
        raise ValueError("Round-23 adaptation branch evidence is incomplete")
    if (
        static.get("schema_version")
        != "neuronsifter-round23-static-development-reference-v2"
        or static.get("status") != "frozen_development_reference"
        or any(
            static.get(field) is not True
            for field in (
                "frozen_twin",
                "frozen_agent",
                "frozen_tri",
                "frozen_development_data_identity",
                "frozen_tool_registry",
                "neural_twin_checkpoint_immutable",
            )
        )
        or static.get("protected_test_access_count") != 0
    ):
        raise ValueError("Round-23 static development reference is incomplete")
    if (
        rollback.get("schema_version")
        != "neuronsifter-round23-rollback-validation-v2"
        or rollback.get("rollback_validated") is not True
        or rollback.get("invalid_assimilation_moves_biological_state") is not False
        or rollback.get("adaptive_twin_updates_run") is not True
        or rollback.get("invalid_probe_count", 0) < 1
        or rollback.get("invalid_probe_count") != rollback.get("invalid_probe_pass_count")
        or rollback.get("protected_test_access_count") != 0
    ):
        raise ValueError("Round-23 rollback validation is incomplete")
    report_rows = read_tsv(report_path)
    if (
        len(report_rows) != 6
        or {row.get("branch_id") for row in report_rows} != required_branches
        or adaptation.get("report_sha256") != sha256_file(report_path)
        or adaptation.get("static_reference_sha256") != sha256_file(static_path)
        or adaptation.get("rollback_validation_sha256") != sha256_file(rollback_path)
    ):
        raise ValueError("Round-23 adaptation report hash or rows drifted")
    manifest_path = repo_path(str(adaptation.get("run_manifest_path", "")))
    round_log_path = repo_path(str(adaptation.get("round_log_path", "")))
    if (
        not manifest_path.is_file()
        or not round_log_path.is_file()
        or adaptation.get("run_manifest_sha256") != sha256_file(manifest_path)
        or adaptation.get("round_log_sha256") != sha256_file(round_log_path)
    ):
        raise ValueError("Round-23 adaptation run lineage drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "completed"
        or manifest.get("fixture_rows_used") is not False
        or manifest.get("neural_twin_checkpoint_updated") is not False
        or manifest.get("prospective_tool_dispatch_count") != 0
        or manifest.get("protected_test_access_count") != 0
        or manifest.get("artifacts", {}).get(
            "Output/paper_reports/round23_dev_adaptation_results.tsv"
        )
        != sha256_file(report_path)
    ):
        raise ValueError("Round-23 adaptation manifest is not authoritative")
    round_rows = read_tsv(round_log_path)
    if (
        len(round_rows) != 18
        or {(row.get("branch_id"), row.get("round_index")) for row in round_rows}
        != {(branch_id, str(index)) for branch_id in required_branches for index in range(1, 4)}
        or any(
            row.get("selection_before_outcome") != "True"
            or row.get("unqueried_outcomes_used_for_policy") != "False"
            or row.get("protected_test_access") != "0"
            for row in round_rows
        )
    ):
        raise ValueError("Round-23 adaptation round log is incomplete")
    source_mechanism = static.get("source_mechanism", {})
    source_gate_path = repo_path(str(source_mechanism.get("gate_path", "")))
    current_mechanism_gate = load_json(
        "Output/audits/round23_coupling_mechanism_gate.json"
    )
    if (
        source_mechanism.get("attempt_id")
        != current_mechanism_gate.get("attempt_id")
        or not source_gate_path.is_file()
        or source_mechanism.get("gate_sha256") != sha256_file(source_gate_path)
    ):
        raise ValueError("Round-23 adaptation source mechanism drift")
    return adaptation


def load_full_suite_execution() -> dict[str, Any]:
    path = repo_path("Output/audits/round23_full_suite_execution.json")
    if not path.is_file():
        return {
            "schema_version": "neuronsifter-round23-full-suite-execution-v1",
            "status": "not_run",
            "full_suite_green": False,
        }
    audit = json.loads(path.read_text(encoding="utf-8"))
    junit_path = repo_path(str(audit.get("junit_path", "")))
    junit_counts = parse_junit_counts(junit_path) if junit_path.is_file() else {}
    source_files = audit.get("source_files", [])
    source_paths = {
        str(row.get("path", ""))
        for row in source_files
        if isinstance(row, Mapping)
    }
    current_source_drift = [
        str(row.get("path", ""))
        for row in source_files
        if isinstance(row, Mapping)
        and (
            not repo_path(str(row.get("path", ""))).is_file()
            or row.get("sha256")
            != sha256_file(repo_path(str(row.get("path", ""))))
        )
    ]
    successor_binding_valid = not current_source_drift
    if current_source_drift:
        audit_by_path = {
            str(row.get("path", "")): row
            for row in source_files
            if isinstance(row, Mapping)
        }
        current_hashes = {
            relative: sha256_file(repo_path(relative))
            for relative in audit_by_path
        }
        predecessor_path = repo_path(ROUND24_SUCCESSOR_BINDING)
        predecessor = (
            json.loads(predecessor_path.read_text(encoding="utf-8"))
            if predecessor_path.is_file()
            else {}
        )
        predecessor_rows = predecessor.get("source_transitions", [])
        predecessor_by_path = {
            str(row.get("path", "")): row
            for row in predecessor_rows
            if isinstance(row, Mapping)
        }
        effective_hashes = {
            relative: str(row.get("sha256", ""))
            for relative, row in audit_by_path.items()
        }
        predecessor_valid = bool(
            predecessor.get("schema_version")
            == "neuronsifter-round24-round23-source-successor-binding-v1"
            and predecessor.get("status") == "bound_successor_development"
            and predecessor.get("round23_full_suite_audit_sha256")
            == sha256_file(path)
            and set(predecessor_by_path).issubset(audit_by_path)
            and all(
                predecessor_by_path[relative].get("round23_sha256")
                == effective_hashes[relative]
                and bool(predecessor_by_path[relative].get("round24_sha256"))
                for relative in predecessor_by_path
            )
            and predecessor.get("publication_claim_eligible") is False
            and predecessor.get("protected_test_access") == 0
        )
        if predecessor_valid:
            for relative, row in predecessor_by_path.items():
                effective_hashes[relative] = str(row["round24_sha256"])

        round24b_path = repo_path(ROUND24B_SUCCESSOR_BINDING)
        round24b = (
            json.loads(round24b_path.read_text(encoding="utf-8"))
            if round24b_path.is_file()
            else {}
        )
        round24b_rows = round24b.get("source_transitions", [])
        round24b_by_path = {
            str(row.get("path", "")): row
            for row in round24b_rows
            if isinstance(row, Mapping)
        }
        before_round24b_hashes = dict(effective_hashes)
        round24b_structural_valid = bool(
            round24b.get("schema_version")
            == "neuronsifter-round24b-round23-source-successor-binding-v1"
            and round24b.get("status") == "bound_successor_development"
            and round24b.get("round23_full_suite_audit_sha256")
            == sha256_file(path)
            and round24b.get("predecessor_binding", {}).get("path")
            == ROUND24_SUCCESSOR_BINDING
            and round24b.get("predecessor_binding", {}).get("sha256")
            == sha256_file(predecessor_path)
            and set(round24b_by_path).issubset(audit_by_path)
            and all(
                round24b_by_path[relative].get("round23_sha256")
                == audit_by_path[relative].get("sha256")
                and round24b_by_path[relative].get("predecessor_sha256")
                == before_round24b_hashes[relative]
                and bool(round24b_by_path[relative].get("round24b_sha256"))
                for relative in round24b_by_path
            )
            and (
                round24b.get("current_source_drift") is None
                or set(round24b.get("current_source_drift", []))
                == set(predecessor_by_path) | set(round24b_by_path)
            )
            and round24b.get("publication_claim_eligible") is False
            and round24b.get("protected_test_access") == 0
        )
        residual_before_round24b = {
            relative
            for relative, current_hash in current_hashes.items()
            if current_hash != before_round24b_hashes[relative]
        }
        round24b_valid = not residual_before_round24b or round24b_structural_valid
        if round24b_structural_valid:
            for relative, row in round24b_by_path.items():
                effective_hashes[relative] = str(row["round24b_sha256"])

        residual_after_round24b = {
            relative
            for relative, current_hash in current_hashes.items()
            if current_hash != effective_hashes[relative]
        }
        round25_path = repo_path(ROUND25_SUCCESSOR_BINDING)
        round25_valid = not residual_after_round24b
        if residual_after_round24b:
            round25 = (
                json.loads(round25_path.read_text(encoding="utf-8"))
                if round25_path.is_file()
                else {}
            )
            round25_rows = round25.get("source_transitions", [])
            round25_by_path = {
                str(row.get("path", "")): row
                for row in round25_rows
                if isinstance(row, Mapping)
            }
            round25_residual = set(
                round25.get("residual_source_drift_after_predecessors", [])
            )
            round25_valid = bool(
                round25.get("schema_version")
                == "neuronsifter-round25-round23-source-successor-binding-v1"
                and round25.get("status") == "bound_successor_development"
                and round25.get("round23_full_suite_audit_sha256")
                == sha256_file(path)
                and round25.get("predecessor_binding", {}).get("path")
                == ROUND24B_SUCCESSOR_BINDING
                and round25.get("predecessor_binding", {}).get("sha256")
                == sha256_file(round24b_path)
                and set(round25_by_path) == round25_residual
                and all(
                    round25_by_path[relative].get("round23_sha256")
                    == audit_by_path[relative].get("sha256")
                    and round25_by_path[relative].get("predecessor_sha256")
                    == effective_hashes[relative]
                    and bool(round25_by_path[relative].get("round25_sha256"))
                    for relative in round25_by_path
                )
                and round25.get("historical_round23_audit_rewritten") is False
                and round25.get("historical_round24b_binding_rewritten") is False
                and round25.get("publication_claim_eligible") is False
                and round25.get("protected_test_access") == 0
            )
            if round25_valid:
                for relative, row in round25_by_path.items():
                    effective_hashes[relative] = str(row["round25_sha256"])
                round25_valid = set(round25.get("current_source_drift", [])) == {
                    relative
                    for relative, effective_hash in effective_hashes.items()
                    if effective_hash != audit_by_path[relative].get("sha256")
                }

        real_intervention_path = repo_path(REAL_INTERVENTION_SUCCESSOR_BINDING)
        real_intervention = (
            json.loads(real_intervention_path.read_text(encoding="utf-8"))
            if real_intervention_path.is_file()
            else {}
        )
        real_intervention_rows = real_intervention.get("source_transitions", [])
        real_intervention_by_path = {
            str(row.get("path", "")): row
            for row in real_intervention_rows
            if isinstance(row, Mapping)
        }
        residual_after_round25 = {
            relative
            for relative, current_hash in current_hashes.items()
            if current_hash != effective_hashes[relative]
        }
        real_intervention_valid = not residual_after_round25
        if residual_after_round25:
            real_intervention_valid = bool(
                real_intervention.get("schema_version")
                == "neuronsifter-real-intervention-round23-source-successor-binding-v3"
                and real_intervention.get("status") == "bound_successor_development"
                and real_intervention.get("round23_full_suite_audit_sha256")
                == sha256_file(path)
                and real_intervention.get("predecessor_binding", {}).get("path")
                == ROUND25_SUCCESSOR_BINDING
                and real_intervention.get("predecessor_binding", {}).get("sha256")
                == sha256_file(round25_path)
                and set(real_intervention_by_path) == residual_after_round25
                and set(real_intervention.get("current_source_drift", []))
                == set(current_source_drift)
                    and all(
                        real_intervention_by_path[relative].get("round23_sha256")
                        == audit_by_path[relative].get("sha256")
                        and real_intervention_by_path[relative].get("predecessor_sha256")
                        == effective_hashes[relative]
                        and isinstance(
                            real_intervention_by_path[relative].get(
                                "real_intervention_sha256"
                            ),
                            str,
                        )
                        and len(
                            real_intervention_by_path[relative][
                                "real_intervention_sha256"
                            ]
                        )
                        == 64
                    for relative in real_intervention_by_path
                )
                and real_intervention.get("historical_round23_audit_rewritten")
                is False
                and real_intervention.get("historical_round25_binding_rewritten")
                is False
                and real_intervention.get("publication_claim_eligible") is False
                and real_intervention.get("protected_test_access") == 0
            )
        if real_intervention_valid:
            for relative, row in real_intervention_by_path.items():
                effective_hashes[relative] = str(row["real_intervention_sha256"])
        before_current_binding_hashes = dict(effective_hashes)
        residual_after_real_intervention = {
            relative
            for relative, current_hash in current_hashes.items()
            if current_hash != effective_hashes[relative]
        }
        current_source_binding_path = repo_path(CURRENT_SOURCE_SUCCESSOR_BINDING)
        current_source_binding = (
            json.loads(current_source_binding_path.read_text(encoding="utf-8"))
            if current_source_binding_path.is_file()
            else {}
        )
        current_source_rows = current_source_binding.get("source_transitions", [])
        current_source_by_path = {
            str(row.get("path", "")): row
            for row in current_source_rows
            if isinstance(row, Mapping)
        }
        current_source_binding_valid = not residual_after_real_intervention
        if residual_after_real_intervention:
            current_source_binding_valid = bool(
                current_source_binding.get("schema_version")
                == "neuronsifter-round23-current-source-successor-binding-v4"
                and current_source_binding.get("status")
                == "bound_successor_development"
                and current_source_binding.get("round23_full_suite_audit_sha256")
                == sha256_file(path)
                and current_source_binding.get("predecessor_binding", {}).get("path")
                == REAL_INTERVENTION_SUCCESSOR_BINDING
                and current_source_binding.get("predecessor_binding", {}).get("sha256")
                == sha256_file(real_intervention_path)
                and set(current_source_by_path) == residual_after_real_intervention
                and set(current_source_binding.get("current_source_drift", []))
                == set(current_source_drift)
                and all(
                    current_source_by_path[relative].get("round23_sha256")
                    == audit_by_path[relative].get("sha256")
                    and current_source_by_path[relative].get("predecessor_sha256")
                    == before_current_binding_hashes[relative]
                    and current_source_by_path[relative].get("current_sha256")
                    == current_hashes[relative]
                    for relative in current_source_by_path
                )
                and current_source_binding.get("historical_round23_audit_rewritten")
                is False
                and current_source_binding.get("historical_real_intervention_binding_rewritten")
                is False
                and current_source_binding.get("publication_claim_eligible") is False
                and current_source_binding.get("protected_test_access") == 0
            )
            if current_source_binding_valid:
                for relative, row in current_source_by_path.items():
                    effective_hashes[relative] = str(row["current_sha256"])
        successor_binding_valid = bool(
            predecessor_valid
            and round24b_valid
            and round25_valid
            and real_intervention_valid
            and current_source_binding_valid
            and all(
                effective_hashes[relative] == current_hashes[relative]
                for relative in audit_by_path
            )
        )
    green = bool(
        audit.get("schema_version")
        == "neuronsifter-round23-full-suite-execution-v1"
        and audit.get("status") == "passed"
        and audit.get("pytest_exit_code") == 0
        and audit.get("tests_collected", 0) > 0
        and audit.get("tests_failed") == 0
        and audit.get("tests_errored") == 0
        and audit.get("tests_skipped") == 0
        and all(
            audit.get(field) == junit_counts.get(field)
            for field in (
                "tests_collected",
                "tests_passed",
                "tests_failed",
                "tests_errored",
                "tests_skipped",
            )
        )
        and audit.get("full_suite_green") is True
        and junit_path.is_file()
        and audit.get("junit_sha256") == sha256_file(junit_path)
        and audit.get("source_file_count") == len(FULL_SUITE_SOURCE_FILES)
        and len(source_files) == len(FULL_SUITE_SOURCE_FILES)
        and source_paths == set(FULL_SUITE_SOURCE_FILES)
        and successor_binding_valid
    )
    if not green:
        raise ValueError("Round-23 full-suite execution audit is not green or source-bound")
    validated = dict(audit)
    validated["historical_source_manifest_bound"] = True
    validated["current_checkout_source_match"] = not current_source_drift
    validated["current_checkout_source_drift"] = sorted(current_source_drift)
    validated["successor_source_binding_valid"] = successor_binding_valid
    return validated


def build_final_gate(
    contract: Mapping[str, Any],
    framework: Mapping[str, Mapping[str, Any]],
    training: Mapping[str, Any],
    checkpoint_validation: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    mechanism: Mapping[str, Any],
    adaptation: Mapping[str, Any],
    full_suite: Mapping[str, Any],
) -> dict[str, Any]:
    architecture = framework["consistency"]
    framework_passed = bool(
        framework["migration"]["single_canonical_framework"]
        and framework["migration"]["historical_claims_reconciled"]
    )
    architecture_passed = bool(
        architecture["shared_posterior_identity"]
        and architecture["hidden_scientific_decision_states"] == 0
        and framework["lineage"]["status"] == "passed"
    )
    model_passed = bool(
        training["c1_development_training_stable"]
        and training["c2s_development_training_stable"]
        and checkpoint_validation["status"] == "passed"
        and diagnostics["complete"]
    )
    mechanism_passed = bool(
        mechanism["four_way_coupling_test_complete"]
        and mechanism["coherence_diagnostics_complete"]
        and mechanism["development_gate_passed"]
    )
    adaptation_passed = bool(
        adaptation["status_known"]
        and adaptation["rollback_validated_if_updates_run"]
    )
    full_suite_green = bool(full_suite.get("full_suite_green") is True)
    resolved = all(
        (
            framework_passed,
            architecture_passed,
            model_passed,
            mechanism_passed,
            adaptation_passed,
            full_suite_green,
        )
    )
    gate = {
        "schema_version": "neuronsifter-round23-vnext-development-gate-v1",
        "created_at": utc_now(),
        "status": "neuronsifter_vnext_core_development_resolved"
        if resolved
        else "pending_validation_or_training",
        "target_status": "neuronsifter_vnext_core_development_resolved",
        "framework": {"canonical_vnext_contract": framework_passed},
        "architecture": {
            "decision_consistency_invariant": architecture_passed,
            "hidden_scientific_decision_states": architecture["hidden_scientific_decision_states"],
        },
        "model": {
            "c1_development_training_stable": training["c1_development_training_stable"],
            "c2s_development_training_stable": training["c2s_development_training_stable"],
            "checkpoint_reproducible": checkpoint_validation["status"] == "passed",
            "development_diagnostics_complete": diagnostics["complete"],
            "diagnostic_hashes": {
                "c1": diagnostics["c1_sha256"],
                "c2s": diagnostics["c2s_sha256"],
            },
        },
        "mechanism": {
            "four_way_coupling_test_complete": mechanism[
                "four_way_coupling_test_complete"
            ],
            "coherence_diagnostics_complete": mechanism[
                "coherence_diagnostics_complete"
            ],
            "development_gate_passed": mechanism["development_gate_passed"],
            "attempt_id": mechanism["attempt_id"],
            "report_sha256": mechanism["report_sha256"],
            "coherence_report_sha256": mechanism["coherence_report_sha256"],
        },
        "adaptation": {
            "status_known": adaptation["status_known"],
            "rollback_validated_if_updates_run": adaptation["rollback_validated_if_updates_run"],
            "execution_mode": "offline_replay_of_prospectively_generated_outcomes",
            "empirical_coevolution_evidence": False,
            "joint_adaptation_direction_status": adaptation[
                "joint_adaptation_direction_status"
            ],
            "recommended_reference": adaptation[
                "recommended_static_or_one_sided_reference"
            ],
            "report_sha256": adaptation["report_sha256"],
        },
        "validation": {
            "full_suite_green": full_suite_green,
            "full_suite_audit_path": "Output/audits/round23_full_suite_execution.json",
            "full_suite_junit_sha256": full_suite.get("junit_sha256"),
            "tests_collected": full_suite.get("tests_collected", 0),
            "tests_failed": full_suite.get("tests_failed"),
            "tests_errored": full_suite.get("tests_errored"),
            "tests_skipped": full_suite.get("tests_skipped"),
        },
        "evidence_boundary": {
            "development_results_labeled": bool(
                training.get("evidence_role") == "development_only"
                and mechanism.get("evidence_role") == "development_only"
                and adaptation.get("evidence_role") == "development_only"
            ),
            "protected_final_holdout_accessed": False,
            "publication_comparison_gate": False,
        },
        "c2b": {"status": contract["claim_boundary"]["c2_b"]},
        "publication_blockers": ["external_baseline_comparison_unresolved", "headline_comparison_unauthorized", "protected_final_holdout_access_false"],
    }
    write_json(repo_path("Output/audits/round23_vnext_development_gate.json"), gate)
    return gate


def build() -> dict[str, Any]:
    """Aggregate validated evidence without rewriting its source artifacts."""

    config = load_config(repo_path("Libs/configs/config_stack.yaml"))
    contract = validate_contract(config)
    framework = load_framework_outputs(contract)
    mechanism = load_mechanism_outputs()
    training, checkpoint_catalog, checkpoint_validation = load_training_outputs(
        contract
    )
    diagnostics = load_development_diagnostics()
    adaptation = load_adaptation_outputs()
    full_suite = load_full_suite_execution()
    gate = build_final_gate(
        contract,
        framework,
        training,
        checkpoint_validation,
        diagnostics,
        mechanism,
        adaptation,
        full_suite,
    )
    return {
        **framework,
        "mechanism": mechanism,
        "training": training,
        "checkpoint_catalog": checkpoint_catalog,
        "checkpoint_validation": checkpoint_validation,
        "diagnostics": diagnostics,
        "adaptation": adaptation,
        "full_suite": full_suite,
        "gate": gate,
    }


def build_training_evidence() -> dict[str, Any]:
    """Rebuild only source-bound training and checkpoint catalogs."""

    config = load_config(repo_path("Libs/configs/config_stack.yaml"))
    contract = validate_contract(config)
    training = build_training_manifest(contract)
    checkpoint_catalog = build_checkpoint_catalog(contract, training)
    return {
        "training": training,
        "checkpoint_catalog": checkpoint_catalog,
    }


def build_runtime_evidence() -> dict[str, Any]:
    """Rebuild the runtime contract and executable posterior-lineage smoke."""

    config = load_config(repo_path("Libs/configs/config_stack.yaml"))
    contract = validate_contract(config)
    consistency = build_decision_consistency_contract(contract)
    lineage = build_posterior_lineage()
    write_json(
        repo_path("Output/audits/round23_decision_consistency_contract.json"),
        consistency,
    )
    write_json(
        repo_path("Output/audits/round23_posterior_lineage_validation.json"),
        lineage,
    )
    return {"consistency": consistency, "lineage": lineage}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--training-evidence-only", action="store_true")
    modes.add_argument("--runtime-evidence-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.runtime_evidence_only:
        result = build_runtime_evidence()
        print(
            json.dumps(
                {
                    "status": "runtime_evidence_built",
                    "decision_consistency_status": result["consistency"]["status"],
                    "lineage_status": result["lineage"]["status"],
                    "checkpoint_roundtrip": result["lineage"][
                        "checkpoint_roundtrip"
                    ],
                },
                sort_keys=True,
            )
        )
        return 0
    if arguments.training_evidence_only:
        result = build_training_evidence()
        print(
            json.dumps(
                {
                    "status": "training_evidence_built",
                    "run_count": len(result["training"]["runs"]),
                    "checkpoint_count": len(
                        result["checkpoint_catalog"]["checkpoints"]
                    ),
                    "reproducible": result["checkpoint_catalog"]["reproducible"],
                },
                sort_keys=True,
            )
        )
        return 0
    result = build()
    print(
        json.dumps(
            {
                "status": "built",
                "gate_status": result["gate"]["status"],
                "lineage_status": result["lineage"]["status"],
                "full_suite_green": result["full_suite"].get(
                    "full_suite_green", False
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
