"""Adjudicate bounded public-evidence scarcity and freeze a prospective protocol."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_retrospective_public_cns_ranking_intake import (  # noqa: E402
    _binding,
)
from Libs.utils.data.raw_data_registry import load_config, write_json, write_text_atomic  # noqa: E402


CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"
SOURCE_DATA_PREREGISTRATION_PATH = (
    REPO_ROOT / "Input/manifests/retrospective_public/source_data_first_preregistration.json"
)
SOURCE_DATA_UNIVERSE_PATH = (
    REPO_ROOT / "Input/manifests/retrospective_public/source_data_metadata_universe.tsv"
)
SOURCE_DATA_QUEUE_PATH = (
    REPO_ROOT / "Input/manifests/retrospective_public/source_data_ranking_queue.tsv"
)
SOURCE_DATA_INTAKE_PATH = (
    REPO_ROOT / "Output/audits/retrospective_public_source_data_first_intake.json"
)
PUBLIC_INTAKE_PATH = REPO_ROOT / "Output/audits/retrospective_public_cns_ranking_intake.json"
PUBLIC_INTAKE_RECOVERY_PATH = (
    REPO_ROOT / "Output/audits/retrospective_public_cns_ranking_source_recovery.json"
)
DESIGN_INTAKE_PATH = (
    REPO_ROOT / "Output/audits/retrospective_public_design_enriched_ranking_intake.json"
)
DESIGN_RECOVERY_PATH = (
    REPO_ROOT / "Output/audits/retrospective_public_design_enriched_ranking_source_recovery.json"
)
MATERIALIZATION_PATH = REPO_ROOT / "Output/audits/retrospective_public_evidence_materialization.json"
REPLAY_PATH = REPO_ROOT / "Output/audits/retrospective_replay_benchmark.json"
LEAKAGE_PATH = REPO_ROOT / "Output/audits/retrospective_replay_leakage_audit.json"

SOURCE_RECOVERY_SCHEMA_PATH = (
    REPO_ROOT / "Libs/configs/schema/retrospective_public_source_data_first_recovery.schema.json"
)
SCARCITY_SCHEMA_PATH = (
    REPO_ROOT
    / "Libs/configs/schema/public_retrospective_ranking_evidence_structural_scarcity.schema.json"
)
PROTOCOL_SCHEMA_PATH = (
    REPO_ROOT
    / "Libs/configs/schema/prospective_uncertainty_aware_validation_protocol.schema.json"
)
YIELD_SCHEMA_PATH = REPO_ROOT / "Libs/configs/schema/public_ranking_evidence_yield.schema.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _payload_binding(path: Path, payload: dict[str, Any]) -> dict[str, object]:
    serialized = (json.dumps(payload, indent=2, ensure_ascii=True) + "\n").encode("utf-8")
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "bytes": len(serialized),
        "sha256": hashlib.sha256(serialized).hexdigest(),
    }


def _schema_errors(payload: dict[str, Any], schema_path: Path) -> list[str]:
    schema = _read_json(schema_path)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: list(error.absolute_path),
    )
    return [f"{list(error.absolute_path)}: {error.message}" for error in errors]


def _load_policy() -> dict[str, Any]:
    config = load_config(CONFIG_PATH)
    policy = config.get("evidence_scarcity_aware_validation")
    if not isinstance(policy, dict):
        raise ValueError("Missing evidence_scarcity_aware_validation configuration")
    return policy


def _validate_current_state() -> dict[str, Any]:
    source_data = _read_json(SOURCE_DATA_INTAKE_PATH)
    public_intake = _read_json(PUBLIC_INTAKE_PATH)
    public_recovery = _read_json(PUBLIC_INTAKE_RECOVERY_PATH)
    design_intake = _read_json(DESIGN_INTAKE_PATH)
    design_recovery = _read_json(DESIGN_RECOVERY_PATH)
    materialization = _read_json(MATERIALIZATION_PATH)
    replay = _read_json(REPLAY_PATH)
    leakage = _read_json(LEAKAGE_PATH)
    if source_data.get("status") != "source_data_queue_exhausted":
        raise ValueError("Source-data-first route is not exhausted")
    if source_data.get("queue_record_count") != 0:
        raise ValueError("Source-data-first queue is not empty")
    if any(item.get("status") != "resolved" for item in source_data.get("repository_family_status", [])):
        raise ValueError("Source-data-first route has unresolved repository families")
    if public_recovery.get("status") != "queue_exhausted_without_new_decision_sensitive_group":
        raise ValueError("Prior bounded publication intake is not terminal")
    if design_recovery.get("status") != "not_run_queue_empty":
        raise ValueError("Design-enriched publication intake is not terminal")
    expected = {
        "status": "qualified_public_candidates_but_split_insufficient",
        "accepted_records": 36,
        "rejected_records": 7,
        "legal_candidate_sets": 6,
        "independent_public_contexts": 6,
        "decision_sensitive_candidate_sets": 2,
        "independent_decision_sensitive_groups": 2,
        "required_groups": 3,
        "split_files_created": False,
        "leakage_status": "passed",
        "protected_holdout_access": 0,
    }
    observed = {
        "status": materialization.get("status"),
        "accepted_records": materialization.get("accepted_record_count"),
        "rejected_records": materialization.get("rejected_record_count"),
        "legal_candidate_sets": materialization.get("legal_candidate_set_count"),
        "independent_public_contexts": materialization.get("independent_context_count"),
        "decision_sensitive_candidate_sets": materialization.get("decision_sensitive_candidate_set_count"),
        "independent_decision_sensitive_groups": replay.get("independent_decision_sensitive_group_count"),
        "required_groups": replay.get("minimum_independent_decision_sensitive_groups"),
        "split_files_created": replay.get("split_files_created"),
        "leakage_status": leakage.get("status"),
        "protected_holdout_access": replay.get("protected_holdout_access"),
    }
    if observed != expected:
        raise ValueError(f"Structural-scarcity parent state drifted: {observed}")
    return {
        "source_data": source_data,
        "public_intake": public_intake,
        "public_recovery": public_recovery,
        "design_intake": design_intake,
        "design_recovery": design_recovery,
        "materialization": materialization,
        "replay": replay,
        "leakage": leakage,
        "current_state": observed,
    }


def _build_source_recovery(state: dict[str, Any], created_at: str) -> dict[str, Any]:
    intake = state["source_data"]
    return {
        "schema_version": "neuronsifter-retrospective-public-source-data-first-recovery-v1",
        "status": "not_run_queue_empty",
        "intake_uid": intake["intake_uid"],
        "preregistration_digest": intake["preregistration_digest"],
        "metadata_universe_digest": intake["metadata_universe_digest"],
        "queue_digest": intake["queue_digest"],
        "repository_family_status": intake["repository_family_status"],
        "metadata_asset_count": intake["metadata_asset_count"],
        "metadata_eligible_count": intake["metadata_eligible_count"],
        "queue_record_count": intake["queue_record_count"],
        "source_recovered_count": 0,
        "formal_abc_admitted_count": 0,
        "new_legal_candidate_count": 0,
        "new_decision_sensitive_group_count": 0,
        "unchanged_artifact_bindings": {
            "evidence_registry": _binding(
                REPO_ROOT / "Input/manifests/retrospective_public/evidence_registry.tsv"
            ),
            "materialization_audit": _binding(MATERIALIZATION_PATH),
            "candidate_registry_and_replay_audit": _binding(REPLAY_PATH),
            "leakage_audit": _binding(LEAKAGE_PATH),
        },
        "final_state": state["current_state"],
        "protected_holdout_access": 0,
        "next_action": "adjudicate_structural_public_ranking_evidence_scarcity",
        "created_at": created_at,
    }


def _route_evidence(state: dict[str, Any]) -> list[dict[str, Any]]:
    public_intake = state["public_intake"]
    public_recovery = state["public_recovery"]
    design_intake = state["design_intake"]
    design_recovery = state["design_recovery"]
    source_data = state["source_data"]
    return [
        {
            "route_uid": "bounded_publication_metadata_intake",
            "discovery_direction": "publication_metadata_first",
            "metadata_records": public_intake["selection_summary"]["metadata_record_count"],
            "metadata_eligible": public_intake["record_count"],
            "queued": public_intake["record_count"],
            "source_recovered": public_recovery["queue_consumed_record_count"],
            "formal_abc": int(bool(public_recovery["formal_evidence_admitted"])),
            "new_decision_sensitive_groups": 0,
            "terminal_status": public_recovery["status"],
        },
        {
            "route_uid": "design_enriched_publication_metadata_intake",
            "discovery_direction": "publication_metadata_first",
            "metadata_records": design_intake["selection_summary"]["metadata_record_count"],
            "metadata_eligible": design_intake["selection_summary"]["eligible_record_count"],
            "queued": design_intake["record_count"],
            "source_recovered": design_recovery["source_recovered_count"],
            "formal_abc": design_recovery["formal_abc_admitted_count"],
            "new_decision_sensitive_groups": design_recovery["new_decision_sensitive_group_count"],
            "terminal_status": design_recovery["status"],
        },
        {
            "route_uid": "public_source_data_first_intake",
            "discovery_direction": "repository_source_data_metadata_first",
            "metadata_records": source_data["metadata_asset_count"],
            "metadata_eligible": source_data["metadata_eligible_count"],
            "queued": source_data["queue_record_count"],
            "source_recovered": 0,
            "formal_abc": 0,
            "new_decision_sensitive_groups": 0,
            "terminal_status": source_data["status"],
        },
    ]


def _build_yield_audit(route_evidence: list[dict[str, Any]], created_at: str) -> dict[str, Any]:
    return {
        "schema_version": "neuronsifter-public-ranking-evidence-yield-v1",
        "status": "bounded_routes_exhausted_no_new_decision_sensitive_group",
        "route_count": len(route_evidence),
        "routes": route_evidence,
        "aggregate_funnel": {
            "publication_metadata_records": sum(
                record["metadata_records"]
                for record in route_evidence
                if record["discovery_direction"] == "publication_metadata_first"
            ),
            "repository_metadata_assets": sum(
                record["metadata_records"]
                for record in route_evidence
                if record["discovery_direction"] == "repository_source_data_metadata_first"
            ),
            "queued_sources": sum(record["queued"] for record in route_evidence),
            "recovered_sources": sum(record["source_recovered"] for record in route_evidence),
            "formal_abc_admissions": sum(record["formal_abc"] for record in route_evidence),
            "new_decision_sensitive_groups": sum(
                record["new_decision_sensitive_groups"] for record in route_evidence
            ),
        },
        "unit_caveat": (
            "Publication metadata records and repository assets are heterogeneous screening units; "
            "their counts are reported separately and are not treated as independent statistical samples."
        ),
        "current_gate": {
            "independent_decision_sensitive_groups": 2,
            "required_groups": 3,
            "remaining_gap": 1,
        },
        "additional_homologous_search_authorized": False,
        "created_at": created_at,
    }


def _build_scarcity_decision(
    state: dict[str, Any],
    source_recovery: dict[str, Any],
    output_path: Path,
    yield_audit: dict[str, Any],
    yield_path: Path,
    created_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": "neuronsifter-public-ranking-evidence-structural-scarcity-v1",
        "status": "PUBLIC_RETROSPECTIVE_THERAPEUTIC_RANKING_EVIDENCE_STRUCTURALLY_SPARSE",
        "scope_boundary": (
            "The decision applies to the completed preregistered public-only publication-metadata "
            "and source-data repository universes under the current A/B/C, therapeutic-action, "
            "shared-control, uncertainty, and decision-sensitivity contracts; it is not a claim "
            "that no qualifying evidence exists anywhere."
        ),
        "route_evidence": _route_evidence(state),
        "current_evidence_state": {
            **state["current_state"],
            "decision_sensitive_context_uids": sorted(
                record["candidate_set_uid"]
                for record in state["replay"]["candidate_sets"]
                if record["decision_classification"] == "decision_sensitive"
            ),
            "ambiguous_context_uids": sorted(
                record["candidate_set_uid"]
                for record in state["replay"]["candidate_sets"]
                if record["decision_classification"] == "ambiguous"
            ),
        },
        "adjudication": {
            "status": "supported_within_bounded_audited_public_search_scope",
            "confidence": "moderate",
            "bottleneck_location": "public_source_discovery_and_quantitative_artifact_availability",
            "statistics_or_threshold_bottleneck": False,
            "additional_homologous_search_recommended": False,
            "threshold_relaxation_recommended": False,
            "ambiguous_context_promotion_allowed": False,
            "global_absence_claimed": False,
        },
        "prohibited_interpretations": [
            "ambiguous contexts are not therapeutic-ranking truth",
            "the two positive contexts do not establish broad CNS generalization",
            "structural scarcity does not authorize retrospective threshold changes",
            "structural scarcity does not authorize model training or protected holdout access",
        ],
        "authorized_next_route": "prospective_uncertainty_aware_protocol_design_only",
        "formal_authority": {
            "additional_evidence_search": False,
            "split": False,
            "world_action": False,
            "gpu": False,
            "tri": False,
            "agent_in_twin": False,
            "coevolution": False,
            "protected_holdout_access": 0,
        },
        "artifact_bindings": {
            "source_data_intake": _binding(SOURCE_DATA_INTAKE_PATH),
            "source_data_recovery": _payload_binding(output_path, source_recovery),
            "ranking_yield_audit": _payload_binding(yield_path, yield_audit),
            "publication_intake": _binding(PUBLIC_INTAKE_PATH),
            "publication_intake_recovery": _binding(PUBLIC_INTAKE_RECOVERY_PATH),
            "design_enriched_intake": _binding(DESIGN_INTAKE_PATH),
            "design_enriched_recovery": _binding(DESIGN_RECOVERY_PATH),
            "materialization": _binding(MATERIALIZATION_PATH),
            "replay": _binding(REPLAY_PATH),
            "leakage": _binding(LEAKAGE_PATH),
        },
        "created_at": created_at,
    }


def _build_protocol(
    policy: dict[str, Any],
    state: dict[str, Any],
    scarcity: dict[str, Any],
    scarcity_path: Path,
    created_at: str,
) -> dict[str, Any]:
    positive = sorted(
        record["candidate_set_uid"]
        for record in state["replay"]["candidate_sets"]
        if record["decision_classification"] == "decision_sensitive"
    )
    ambiguous = sorted(
        record["candidate_set_uid"]
        for record in state["replay"]["candidate_sets"]
        if record["decision_classification"] == "ambiguous"
    )
    evaluation_population = {
        **policy["evaluation_population"],
        "positive_ranking_context_count": len(positive),
        "positive_ranking_context_uids": positive,
        "ambiguous_context_count": len(ambiguous),
        "ambiguous_context_uids": ambiguous,
        "total_context_count": len(positive) + len(ambiguous),
        "train_context_count": 0,
        "calibration_context_count": 0,
    }
    return {
        "schema_version": "neuronsifter-prospective-uncertainty-aware-validation-protocol-v1",
        "status": "protocol_frozen_execution_not_authorized",
        "protocol_uid": "evidence_scarcity_aware_agent_in_twin_validation",
        "claim_boundaries": policy["claim_boundaries"],
        "evaluation_population": evaluation_population,
        "model_route_comparison": policy["model_route_comparison"],
        "tasks": policy["tasks"],
        "allowed_information_actions": policy["allowed_information_actions"],
        "baselines": policy["baselines"],
        "metrics": policy["metrics"],
        "reporting_policy": policy["reporting_policy"],
        "acceptance_conditions": policy["acceptance_conditions"],
        "execution_authority": policy["execution_authority"],
        "evidence_bindings": {
            "structural_scarcity_decision": _payload_binding(scarcity_path, scarcity),
            "materialization": _binding(MATERIALIZATION_PATH),
            "replay": _binding(REPLAY_PATH),
            "agent_in_twin_engineering_closure": _binding(
                REPO_ROOT / "Output/audits/agent_in_twin_engineering_closure.json"
            ),
        },
        "created_at": created_at,
    }


def _failure_analysis(
    state: dict[str, Any],
    source_recovery: dict[str, Any],
    scarcity: dict[str, Any],
    protocol: dict[str, Any],
) -> str:
    route_rows = "\n".join(
        "| {route_uid} | {discovery_direction} | {metadata_records} | {metadata_eligible} | "
        "{queued} | {source_recovered} | {formal_abc} | {new_decision_sensitive_groups} | {terminal_status} |".format(
            **record
        )
        for record in scarcity["route_evidence"]
    )
    family_rows = "\n".join(
        f"| {record['family_uid']} | {record['status']} | {record['metadata_asset_count']} | {record['asset_cap']} |"
        for record in source_recovery["repository_family_status"]
    )
    return f"""# Evidence-Scarcity Failure Analysis

## Decision

`{scarcity['status']}`

This decision is limited to the completed preregistered public-only search universes. It does not claim that qualifying evidence is globally absent.

## Current Evidence State

- Accepted formal records: {state['current_state']['accepted_records']}
- Rejected records: {state['current_state']['rejected_records']}
- Legal candidate sets: {state['current_state']['legal_candidate_sets']}
- Independent public contexts: {state['current_state']['independent_public_contexts']}
- Independent decision-sensitive groups: {state['current_state']['independent_decision_sensitive_groups']} / {state['current_state']['required_groups']}
- Split files created: {str(state['current_state']['split_files_created']).lower()}
- Leakage audit: {state['current_state']['leakage_status']}
- Protected holdout access: {state['current_state']['protected_holdout_access']}

## Bounded Route Evidence

| Route | Discovery direction | Metadata records/assets | Metadata eligible | Queued | Recovered | Formal A/B/C | New decision-sensitive groups | Terminal status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
{route_rows}

## Source-Data Repository Coverage

| Repository family | Status | Assets screened | Cap |
| --- | --- | ---: | ---: |
{family_rows}

## Root-Cause Assessment

The dominant bottleneck is upstream evidence availability and metadata eligibility, not the frozen bootstrap threshold or candidate-ranking statistic. The source-data-first route resolved all preregistered repository families but produced no asset satisfying the joint requirements for CNS disease or injury context, structured therapeutic actions, shared-control evidence, functional endpoints, and biological-replication or uncertainty metadata.

The result does not justify relaxing A/B/C admission, changing endpoints, treating technical replicates as biological replicates, promoting toxicant controls, or converting ambiguous contexts into ranking truth. Additional homologous publication or repository searches are closed unless a materially different external evidence source is explicitly authorized.

## Prospective Route

The old retrospective ranking claim remains unresolved at 2/3. A separate prospective protocol is frozen for uncertainty-aware evaluation using {len(protocol['evaluation_population']['positive_ranking_context_uids'])} decision-sensitive contexts as ranking evaluation cases and {len(protocol['evaluation_population']['ambiguous_context_uids'])} ambiguous contexts as abstention and information-acquisition cases. None of these contexts may be used for training or threshold fitting.

Preferred future evaluation route: `{protocol['model_route_comparison']['preferred_route']}`. Execution remains unauthorized.
"""


def build(*, write_output: bool = True) -> dict[str, dict[str, Any] | str]:
    policy = _load_policy()
    outputs = {
        key: (REPO_ROOT / str(value)).resolve()
        for key, value in policy["outputs"].items()
    }
    state = _validate_current_state()
    created_at = _now()
    source_recovery = _build_source_recovery(state, created_at)
    source_errors = _schema_errors(source_recovery, SOURCE_RECOVERY_SCHEMA_PATH)
    if source_errors:
        raise ValueError("Source-data recovery schema errors: " + "; ".join(source_errors[:8]))
    route_evidence = _route_evidence(state)
    yield_audit = _build_yield_audit(route_evidence, created_at)
    yield_errors = _schema_errors(yield_audit, YIELD_SCHEMA_PATH)
    if yield_errors:
        raise ValueError("Ranking yield schema errors: " + "; ".join(yield_errors[:8]))
    scarcity = _build_scarcity_decision(
        state,
        source_recovery,
        outputs["source_data_recovery_audit"],
        yield_audit,
        outputs["ranking_yield_audit"],
        created_at,
    )
    scarcity_errors = _schema_errors(scarcity, SCARCITY_SCHEMA_PATH)
    if scarcity_errors:
        raise ValueError("Structural scarcity schema errors: " + "; ".join(scarcity_errors[:8]))
    protocol = _build_protocol(
        policy,
        state,
        scarcity,
        outputs["structural_scarcity_decision"],
        created_at,
    )
    protocol_errors = _schema_errors(protocol, PROTOCOL_SCHEMA_PATH)
    if protocol_errors:
        raise ValueError("Prospective protocol schema errors: " + "; ".join(protocol_errors[:8]))
    report = _failure_analysis(state, source_recovery, scarcity, protocol)
    if write_output:
        write_json(outputs["source_data_recovery_audit"], source_recovery)
        write_json(outputs["ranking_yield_audit"], yield_audit)
        write_json(outputs["structural_scarcity_decision"], scarcity)
        write_json(outputs["prospective_protocol_audit"], protocol)
        write_text_atomic(outputs["failure_analysis_report"], report)
    return {
        "source_data_recovery": source_recovery,
        "ranking_yield": yield_audit,
        "structural_scarcity": scarcity,
        "prospective_protocol": protocol,
        "failure_analysis_report": report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    result = build(write_output=not args.no_write)
    print(
        json.dumps(
            {
                "source_data_recovery_status": result["source_data_recovery"]["status"],
                "structural_scarcity_status": result["structural_scarcity"]["status"],
                "prospective_protocol_status": result["prospective_protocol"]["status"],
                "current_independent_decision_sensitive_groups": result["structural_scarcity"]["current_evidence_state"]["independent_decision_sensitive_groups"],
                "required_groups": result["structural_scarcity"]["current_evidence_state"]["required_groups"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
