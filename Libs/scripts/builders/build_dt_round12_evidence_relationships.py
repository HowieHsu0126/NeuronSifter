"""Build the read-only Round-12 typed evidence relationship pilot."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

from jsonschema import Draft202012Validator

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.evidence_relationships import (  # noqa: E402
    IDENTITY_AXES,
    PAIRING_REQUIRED_AXES,
    SCHEMA_VERSIONS,
    SUPPORT_DOMAIN_FIELDS,
    EvidenceConflict,
    EvidenceNode,
    EvidencePairingDecision,
    EvidenceRelation,
    EvidenceRelationProvenance,
    EvidenceSupportDomain,
    derive_eligibility,
    frozen_gap_rows,
    round12_gate_status,
    stable_uid,
    validate_frozen_gap_preservation,
    validate_graph,
)
from Libs.utils.data.evidence_union import build_evidence_union  # noqa: E402
from Libs.utils.data.raw_data_registry import load_config, sha256_of_file, write_json, write_tsv  # noqa: E402
from Libs.utils.exp.orchestration_controller import (  # noqa: E402
    CANONICAL_CONTROLLER_UID,
    build_canonical_gate_decision,
    git_commit,
    load_authority_contract,
    sha256_file,
    sha256_payload,
)


OUTPUT_PATHS = {
    "graph": "Output/audits/dt_round12_relationship_graph.json",
    "nodes": "Output/audits/dt_round12_relationship_node_catalog.tsv",
    "relations": "Output/audits/dt_round12_relationship_edge_catalog.tsv",
    "pairings": "Output/audits/dt_round12_pairing_decision_catalog.tsv",
    "domains": "Output/audits/dt_round12_support_domain_catalog.tsv",
    "eligibilities": "Output/audits/dt_round12_eligibility_catalog.tsv",
    "conflicts": "Output/audits/dt_round12_conflict_catalog.tsv",
    "gaps": "Output/audits/dt_round12_frozen_gap_relationship_map.tsv",
    "validation": "Output/audits/dt_round12_validation.json",
    "gate": "Output/audits/dt_round12_gate.json",
}

SCHEMA_PATHS = {
    "node": "Libs/configs/schema/evidence_node.schema.json",
    "relation": "Libs/configs/schema/evidence_relation.schema.json",
    "pairing": "Libs/configs/schema/evidence_pairing_decision.schema.json",
    "support_domain": "Libs/configs/schema/evidence_support_domain.schema.json",
    "eligibility": "Libs/configs/schema/evidence_eligibility.schema.json",
    "conflict": "Libs/configs/schema/evidence_conflict.schema.json",
    "graph": "Libs/configs/schema/evidence_relationship_graph.schema.json",
}
AUTHORITY_PATH = "Libs/configs/validation/round11f_controller_authority.yaml"


def _identity(**values: object) -> dict[str, object]:
    return {axis: values[axis] for axis in IDENTITY_AXES if axis in values}


def _axis_results(**values: str) -> dict[str, str]:
    result = {axis: "not_applicable" for axis in IDENTITY_AXES}
    result.update(values)
    return result


def _bundle_asset(bundle: Mapping[str, object], logical_roles: Sequence[str]) -> dict[str, object]:
    entries = [
        dict(value)
        for group in ("processed_input_entries", "external_asset_entries")
        for value in bundle.get(group, [])
        if isinstance(value, Mapping)
    ]
    for logical_role in logical_roles:
        for entry in entries:
            if entry.get("logical_role") == logical_role:
                return entry
    if entries:
        return entries[0]
    raise ValueError("Execution input bundle has no authorized asset entries.")


def _provenance(uid: str, asset: Mapping[str, object], *, status: str = "verified") -> dict[str, object]:
    asset_uid = str(asset.get("asset_uid", ""))
    content_hash = str(asset.get("content_hash", ""))
    return EvidenceRelationProvenance(
        schema_version=SCHEMA_VERSIONS["provenance"],
        provenance_uid=uid,
        source_artifact_uids=[asset_uid],
        producer_uid=str(asset.get("producer_uid", "")) if status == "verified" else "unverified_source",
        source_schema_versions=[str(asset.get("version", ""))],
        content_hashes={asset_uid: content_hash},
        provenance_status=status,
        source_uri=str(asset.get("canonical_uri", "repo://Output/audits/unverified")),
        status="pilot_only" if status == "verified" else "rejected",
    ).to_dict()


def _node(
    uid: str,
    node_type: str,
    evidence_class: str,
    provenance_ref: str,
    *,
    qc_grade: str = "not_applicable",
    status: str = "pilot_only",
    **axes: object,
) -> dict[str, object]:
    return EvidenceNode(
        schema_version=SCHEMA_VERSIONS["node"],
        node_uid=uid,
        node_type=node_type,
        evidence_class=evidence_class,
        qc_grade=qc_grade,
        identity_axes=_identity(**axes),
        provenance_ref=provenance_ref,
        train_eval_role="round12_contract_validation_only",
        status=status,
    ).to_dict()


def _pairing(
    source: str,
    target: str,
    decision: str,
    axis_results: Mapping[str, str],
    *,
    evidence: Mapping[str, object] | None = None,
    blockers: Sequence[str] = (),
    status: str = "pilot_only",
) -> dict[str, object]:
    return EvidencePairingDecision(
        schema_version=SCHEMA_VERSIONS["pairing"],
        pairing_uid=stable_uid("pairing", {"source": source, "target": target, "decision": decision}),
        source_node_uid=source,
        target_node_uid=target,
        decision=decision,
        identity_axis_results=dict(axis_results),
        required_identity_axes=sorted(PAIRING_REQUIRED_AXES.get(decision, ())),
        pairing_evidence=dict(evidence or {}),
        blockers=list(blockers),
        status=status,
    ).to_dict()


def _domain(uid: str, provenance_ref: str, *, generalization_status: str = "in_domain") -> dict[str, object]:
    scopes = {field: ["not_applicable"] for field in SUPPORT_DOMAIN_FIELDS}
    scopes.update(
        {
            "organism_scope": ["human"],
            "disease_scope": ["ad_control"],
            "region_scope": ["hippocampus"],
            "compartment_scope": ["isf"],
            "cell_type_scope": ["mixed_cns"],
            "modality_scope": ["transcriptomics"],
            "assay_scope": ["expression_profiling"],
            "source_scope": ["round11g_authorized_execution_input_bundle"],
        }
    )
    return EvidenceSupportDomain(
        schema_version=SCHEMA_VERSIONS["support_domain"],
        support_domain_uid=uid,
        generalization_status=generalization_status,
        provenance_ref=provenance_ref,
        status="pilot_only" if generalization_status == "in_domain" else "stress_only" if generalization_status == "stress_only" else "blocked",
        **scopes,
    ).to_dict()


def _relation_spec(
    relation_type: str,
    source: str,
    target: str,
    domain_ref: str,
    provenance_ref: str,
    *,
    biological: str = "not_applicable",
    temporal: str = "not_applicable",
    action: str = "not_applicable",
    context: str = "exact",
    status: str = "pilot_only",
    confidence: float = 1.0,
) -> dict[str, object]:
    uid = stable_uid("relation", {"type": relation_type, "source": source, "target": target})
    return {
        "schema_version": SCHEMA_VERSIONS["relation"],
        "relation_uid": uid,
        "relation_type": relation_type,
        "source_node_uid": source,
        "target_node_uid": target,
        "directionality": "bidirectional" if relation_type in {"conflicts_with", "duplicates"} else "directed",
        "biological_unit_relation": biological,
        "temporal_relation": temporal,
        "action_relation": action,
        "context_relation": context,
        "support_domain_ref": domain_ref,
        "eligibility_ref": "",
        "provenance_ref": provenance_ref,
        "confidence": confidence,
        "status": status,
    }


def build_pilot_graph(
    execution_bundle: Mapping[str, object],
    *,
    bundle_sha256: str,
) -> dict[str, object]:
    action_asset = _bundle_asset(execution_bundle, ("action_ready", "intervention_program"))
    simulation_asset = _bundle_asset(execution_bundle, ("simulation_audit", "simulation_teacher_ready"))
    observation_asset = _bundle_asset(execution_bundle, ("model_input", "analysis_ready"))

    provenances = [
        _provenance("prov_observation_bundle", observation_asset),
        _provenance("prov_action_bundle", action_asset),
        _provenance("prov_simulation_bundle", simulation_asset),
        _provenance(
            "prov_unknown_rejected",
            {"asset_uid": "unknown-source", "content_hash": "0" * 64, "version": "unknown", "canonical_uri": "repo://Output/audits/unverified"},
            status="unknown",
        ),
    ]
    domains = [
        _domain("domain_human_ad_hippocampus", "prov_observation_bundle"),
        _domain("domain_simulator_stress", "prov_simulation_bundle", generalization_status="stress_only"),
        _domain("domain_hypothetical_unsupported", "prov_action_bundle", generalization_status="unsupported"),
    ]

    common_axes = {
        "publication_uid": "pilot_publication",
        "accession_uid": "pilot_accession",
        "study_uid": "pilot_study",
        "cohort_uid": "pilot_cohort",
        "biological_unit_uid": "pilot_biological_unit",
        "organism": "human",
        "disease_state": "ad_control",
        "tissue": "brain",
        "brain_region": "hippocampus",
        "compartment": "isf",
        "cell_type": "mixed_cns",
        "modality": "transcriptomics",
        "assay": "expression_profiling",
    }
    nodes = [
        _node("node_publication", "publication", "contract_validation_example", "prov_observation_bundle", publication_uid="pilot_publication"),
        _node("node_accession", "accession", "contract_validation_example", "prov_observation_bundle", publication_uid="pilot_publication", accession_uid="pilot_accession"),
        _node("node_study", "study", "contract_validation_example", "prov_observation_bundle", **common_axes),
        _node("node_biological_unit", "biological_unit", "contract_validation_example", "prov_observation_bundle", **common_axes),
        _node("node_observation_baseline", "observation", "contract_validation_example", "prov_observation_bundle", **common_axes, sample_uid="sample_baseline", timepoint="t0"),
        _node("node_observation_post", "observation", "contract_validation_example", "prov_observation_bundle", **common_axes, sample_uid="sample_post", timepoint="t1", action_uid="pilot_action", dose="1 unit", route="in_vitro", schedule="single"),
        _node("node_observation_cross", "observation", "contract_validation_example", "prov_observation_bundle", **{**common_axes, "biological_unit_uid": "pilot_biological_unit_cross"}, sample_uid="sample_cross", timepoint="t0"),
        _node("node_state_snapshot", "state_snapshot", "contract_validation_example", "prov_observation_bundle", **common_axes, timepoint="t0"),
        _node("node_action", "structured_action", "contract_validation_example", "prov_action_bundle", **common_axes, action_uid="pilot_action", target_program_uid="pilot_target_program", dose="1 unit", route="in_vitro", schedule="single"),
        _node("node_endpoint", "perturbation_endpoint", "contract_validation_example", "prov_action_bundle", **common_axes, sample_uid="sample_post", action_uid="pilot_action", dose="1 unit", route="in_vitro", schedule="single", timepoint="t1"),
        _node("node_endpoint_conflict", "perturbation_endpoint", "contract_validation_example", "prov_action_bundle", **common_axes, sample_uid="sample_post_conflict", action_uid="pilot_action", dose="1 unit", route="in_vitro", schedule="single", timepoint="t1"),
        _node("node_mechanistic_model", "mechanistic_model", "contract_validation_example", "prov_simulation_bundle", simulator_family_uid="pilot_model_family"),
        _node("node_simulator_output", "simulator_output", "contract_validation_example", "prov_simulation_bundle", action_uid="pilot_action", trajectory_family_uid="pilot_trajectory_family", simulator_family_uid="pilot_model_family", timepoint="series"),
        _node("node_trajectory", "trajectory", "contract_validation_example", "prov_simulation_bundle", action_uid="pilot_action", trajectory_family_uid="pilot_trajectory_family", simulator_family_uid="pilot_model_family", timepoint="series"),
        _node("node_tool_request", "tool_request", "contract_validation_example", "prov_simulation_bundle", tool_family_uid="pilot_tool_family"),
        _node("node_tool_result", "tool_result", "contract_validation_example", "prov_simulation_bundle", tool_family_uid="pilot_tool_family"),
        _node("node_hypothetical", "hypothetical_scenario", "hypothetical", "prov_action_bundle", action_uid="hypothetical_action"),
        _node("node_claim", "claim", "semantic_claim", "prov_observation_bundle", publication_uid="pilot_publication"),
        _node("node_unknown_observation", "observation", "contract_validation_example", "prov_unknown_rejected", status="rejected", biological_unit_uid=None),
    ]

    exact_observation_axes = _axis_results(
        publication_uid="exact_match", accession_uid="exact_match", study_uid="exact_match", cohort_uid="exact_match",
        biological_unit_uid="exact_match", sample_uid="different_but_allowed", organism="exact_match", disease_state="exact_match",
        tissue="exact_match", brain_region="exact_match", compartment="exact_match", cell_type="exact_match",
        modality="exact_match", assay="exact_match", timepoint="different_but_allowed",
    )
    pairings = [
        _pairing("node_observation_baseline", "node_observation_post", "valid_longitudinal_pair", exact_observation_axes),
        _pairing(
            "node_observation_baseline",
            "node_endpoint",
            "valid_action_endpoint_pair",
            {**exact_observation_axes, "action_uid": "exact_match", "target_program_uid": "exact_match", "dose": "exact_match", "route": "exact_match", "schedule": "exact_match"},
            evidence={
                "baseline_identity": "node_observation_baseline", "post_action_identity": "node_endpoint",
                "biological_unit_relationship": "same_unit", "action_identity": "pilot_action", "dose": "1 unit",
                "route_or_context": "in_vitro", "time_relationship": "before_after",
                "measurement_compatibility": "same_assay", "source_provenance": "prov_action_bundle",
            },
        ),
        _pairing(
            "node_observation_cross", "node_state_snapshot", "valid_cross_sectional_comparison",
            _axis_results(study_uid="exact_match", biological_unit_uid="different_but_allowed", organism="exact_match", disease_state="exact_match", tissue="exact_match", modality="exact_match", assay="exact_match"),
        ),
        _pairing(
            "node_simulator_output", "node_trajectory", "valid_simulator_trajectory_pair",
            _axis_results(action_uid="exact_match", timepoint="compatible", trajectory_family_uid="exact_match", simulator_family_uid="exact_match"),
        ),
        _pairing(
            "node_mechanistic_model", "node_simulator_output", "valid_model_output_pair",
            _axis_results(timepoint="compatible", simulator_family_uid="exact_match"),
        ),
        _pairing("node_observation_cross", "node_claim", "semantic_alignment_only", _axis_results()),
        _pairing("node_hypothetical", "node_claim", "unsupported_pairing", _axis_results(), blockers=("hypothetical_not_empirical_evidence",), status="rejected"),
        _pairing("node_unknown_observation", "node_claim", "insufficient_identity", _axis_results(biological_unit_uid="unknown"), blockers=("unknown_provenance", "missing_biological_unit_uid"), status="rejected"),
    ]

    relation_specs = [
        _relation_spec("publication_reports_observation", "node_publication", "node_observation_baseline", "domain_human_ad_hippocampus", "prov_observation_bundle"),
        _relation_spec("accession_contains_observation", "node_accession", "node_observation_baseline", "domain_human_ad_hippocampus", "prov_observation_bundle"),
        _relation_spec("study_contains_biological_unit", "node_study", "node_biological_unit", "domain_human_ad_hippocampus", "prov_observation_bundle"),
        _relation_spec("same_unit_longitudinal_observation", "node_observation_baseline", "node_observation_post", "domain_human_ad_hippocampus", "prov_observation_bundle", biological="same_unit", temporal="before_after"),
        _relation_spec("same_unit_action_endpoint", "node_observation_baseline", "node_endpoint", "domain_human_ad_hippocampus", "prov_action_bundle", biological="same_unit", temporal="before_after", action="action_to_endpoint"),
        _relation_spec("cross_sectional_state_observation", "node_observation_cross", "node_state_snapshot", "domain_human_ad_hippocampus", "prov_observation_bundle", biological="different_unit", temporal="cross_sectional"),
        _relation_spec("static_perturbation_endpoint", "node_action", "node_endpoint", "domain_human_ad_hippocampus", "prov_action_bundle", biological="same_unit", temporal="static_endpoint", action="action_to_endpoint"),
        _relation_spec("simulator_generated_trajectory", "node_simulator_output", "node_trajectory", "domain_simulator_stress", "prov_simulation_bundle", temporal="simulated_time", action="same_action", context="stress_only", status="stress_only"),
        _relation_spec("published_mechanistic_model_output", "node_mechanistic_model", "node_simulator_output", "domain_simulator_stress", "prov_simulation_bundle", temporal="simulated_time", context="stress_only", status="stress_only"),
        _relation_spec("tool_derived_evidence", "node_tool_request", "node_tool_result", "domain_simulator_stress", "prov_simulation_bundle", context="stress_only", status="stress_only"),
        _relation_spec("semantic_alignment_only", "node_observation_cross", "node_claim", "domain_human_ad_hippocampus", "prov_observation_bundle", context="different_but_allowed"),
        _relation_spec("hypothetical_action_scenario", "node_hypothetical", "node_claim", "domain_hypothetical_unsupported", "prov_action_bundle", action="hypothetical_action", context="unsupported", status="rejected", confidence=0.0),
        _relation_spec("conflicts_with", "node_endpoint", "node_endpoint_conflict", "domain_human_ad_hippocampus", "prov_action_bundle", biological="same_unit", temporal="same_timepoint", action="same_action", status="stress_only", confidence=0.5),
        _relation_spec("supports_claim", "node_observation_post", "node_claim", "domain_human_ad_hippocampus", "prov_observation_bundle", biological="same_unit", temporal="static_endpoint"),
        _relation_spec("supports_claim", "node_unknown_observation", "node_claim", "domain_human_ad_hippocampus", "prov_unknown_rejected", context="unknown", status="rejected", confidence=0.0),
    ]
    conflict_relation_uid = next(row["relation_uid"] for row in relation_specs if row["relation_type"] == "conflicts_with")
    endpoint_relation_uid = next(row["relation_uid"] for row in relation_specs if row["relation_type"] == "same_unit_action_endpoint")
    conflicts = [
        EvidenceConflict(
            schema_version=SCHEMA_VERSIONS["conflict"],
            conflict_uid="conflict_directional_pilot",
            conflict_type="directional_disagreement",
            member_uids=[endpoint_relation_uid, conflict_relation_uid],
            state="unresolved",
            typed_context_difference={},
            provenance_ref="prov_action_bundle",
            status="pilot_only",
        ).to_dict()
    ]

    node_map = {row["node_uid"]: row for row in nodes}
    pairing_map = {(row["source_node_uid"], row["target_node_uid"]): row for row in pairings}
    domain_map = {row["support_domain_uid"]: row for row in domains}
    provenance_map = {row["provenance_uid"]: row for row in provenances}
    relations: list[dict[str, object]] = []
    eligibilities: list[dict[str, object]] = []
    for spec in relation_specs:
        relation_conflicts = [row for row in conflicts if spec["relation_uid"] in row["member_uids"]]
        eligibility = derive_eligibility(
            subject_uid=str(spec["relation_uid"]),
            node=node_map[str(spec["target_node_uid"])],
            relation=spec,
            pairing=pairing_map.get((str(spec["source_node_uid"]), str(spec["target_node_uid"]))),
            domain=domain_map[str(spec["support_domain_ref"])],
            provenance=provenance_map[str(spec["provenance_ref"])],
            conflicts=relation_conflicts,
        )
        spec["eligibility_ref"] = eligibility["eligibility_uid"]
        relations.append(EvidenceRelation(**spec).to_dict())
        eligibilities.append(eligibility)

    member_base = {
        "producer_uid": "round12_pilot_builder",
        "schema_version": SCHEMA_VERSIONS["relation"],
        "lineage_group": "round12_pilot_endpoint",
        "provenance_status": "verified",
        "evidence_type": "contract_validation_example",
        "evidence_class": "contract_validation_example",
        "qc_grade": "not_applicable",
        "train_eval_role": "contract_validation_only",
        "claim_eligibility": "blocked",
        "independence_group": "round12_pilot_endpoint",
        "status": "accepted",
        "semantic_fingerprint": "round12_pilot_endpoint",
        "pairing_decision": "valid_action_endpoint_pair",
        "typed_conflicts": conflicts,
    }
    union = build_evidence_union(
        union_uid="round12_pilot_conflict_preserving_union",
        members=[
            {**member_base, "artifact_uid": "pilot_endpoint_left", "artifact_hash": "1" * 64, "relationship_eligibility": {key: False for key in eligibilities[0] if key.startswith("eligible_for_")}},
            {**member_base, "artifact_uid": "pilot_endpoint_right", "artifact_hash": "2" * 64, "relationship_eligibility": {key: False for key in eligibilities[0] if key.startswith("eligible_for_")}},
        ],
        canonical_output_ref="repo://Output/audits/dt_round12_relationship_graph.json",
    )

    graph_payload = {
        "schema_version": SCHEMA_VERSIONS["graph"],
        "graph_uid": stable_uid("round12_graph", {"bundle_sha256": bundle_sha256, "node_uids": [row["node_uid"] for row in nodes]}),
        "contract_scope": "round12_contract_validation_pilot_only",
        "source_bundle": {
            "canonical_uri": "repo://Output/audits/dt_round11g1_execution_input_bundle.json",
            "sha256": bundle_sha256,
            "code_commit": str(execution_bundle.get("code_commit", "")),
            "validation_status": str(execution_bundle.get("validation_status", "")),
        },
        "nodes": nodes,
        "relations": relations,
        "support_domains": domains,
        "eligibilities": eligibilities,
        "conflicts": conflicts,
        "pairing_decisions": pairings,
        "provenances": provenances,
        "evidence_unions": [union],
        "evidence_fabric_sealed": False,
        "dt_core": "not_evaluated",
    }
    return graph_payload


def _schema_blockers(graph: Mapping[str, object], repository_root: Path) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    groups = {
        "node": graph["nodes"], "relation": graph["relations"], "pairing": graph["pairing_decisions"],
        "support_domain": graph["support_domains"], "eligibility": graph["eligibilities"], "conflict": graph["conflicts"],
        "graph": [graph],
    }
    for family, values in groups.items():
        schema = json.loads((repository_root / SCHEMA_PATHS[family]).read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        for index, value in enumerate(values):
            for error in validator.iter_errors(value):
                blockers.append({"code": "json_schema_validation", "message": error.message, "schema_family": family, "index": index, "path": list(error.path)})
    return blockers


def build_round12_artifacts(
    *,
    execution_bundle_path: Path,
    expected_bundle_sha256: str,
    frozen_validation_path: Path,
    expected_code_commit: str,
    repository_root: Path = REPO_ROOT,
    execution_dirty_status: str = "dirty",
) -> dict[str, object]:
    observed_bundle_sha256 = sha256_of_file(execution_bundle_path)
    bundle = json.loads(execution_bundle_path.read_text(encoding="utf-8"))
    frozen_validation = json.loads(frozen_validation_path.read_text(encoding="utf-8"))
    config = load_config(repository_root / "Libs/configs/config_stack.yaml")
    source_adapters = config["digital_twin_evidence_fabric"]["source_adapters"]
    preflight_blockers: list[dict[str, object]] = []
    if observed_bundle_sha256 != expected_bundle_sha256:
        preflight_blockers.append({"code": "execution_bundle_hash_mismatch", "message": "Execution input bundle SHA-256 does not match the frozen Round-11 identity.", "expected": expected_bundle_sha256, "actual": observed_bundle_sha256})
    if bundle.get("code_commit") != expected_code_commit or bundle.get("validation_status") != "passed" or bundle.get("missing_entries") != []:
        preflight_blockers.append({"code": "execution_bundle_identity_mismatch", "message": "Execution input bundle does not preserve the accepted Round-11 code/input identity."})

    graph = build_pilot_graph(bundle, bundle_sha256=observed_bundle_sha256)
    validation = validate_graph(graph)
    validation["blockers"].extend(_schema_blockers(graph, repository_root))
    gaps = frozen_gap_rows(frozen_validation, source_adapters)
    gap_blockers = validate_frozen_gap_preservation(frozen_validation, gaps, source_adapters)
    validation["blockers"].extend(preflight_blockers + gap_blockers)
    validation["blockers"] = sorted(validation["blockers"], key=lambda row: (str(row.get("code", "")), json.dumps(row, sort_keys=True)))
    validation["blocker_count"] = len(validation["blockers"])
    validation["status"] = "passed" if not validation["blockers"] else "blocked"
    validation.update(
        {
            "round_uid": "round_12_typed_evidence_relationships",
            "source_bundle_sha256": observed_bundle_sha256,
            "source_code_commit": str(bundle.get("code_commit", "")),
            "frozen_gap_source_count": 20,
            "frozen_gap_mapped_count": len(gaps),
            "frozen_gap_count_preserved": len(gaps) == 20 and not gap_blockers,
            "pilot_scope": "contract_validation_only_not_evidence_fabric",
            "mandatory_negative_case_count": 15,
            "mandatory_negative_cases_verified": False,
            "full_suite_status": "not_verified",
            "execution_dirty_status": execution_dirty_status,
            "execution_identity_relation": "not_exact_clean_commit" if execution_dirty_status != "clean" else "exact_clean_commit",
        }
    )
    gate_blockers = deepcopy(validation["blockers"])
    if execution_dirty_status != "clean":
        gate_blockers.append({"code": "execution_identity_not_exact_clean", "message": "Round-12 Gate requires a clean exact execution commit."})
    gate_blockers.append({"code": "full_suite_not_verified", "message": "Round-12 requires a commit-bound zero-failure full-suite result."})
    gate_status = round12_gate_status(gate_blockers)
    authority = load_authority_contract(repository_root / AUTHORITY_PATH)
    gate = build_canonical_gate_decision(
        authority,
        gate_uid="round_12_typed_evidence_relationships",
        stage_uid="round_12_typed_evidence_relationships",
        controller_uid=CANONICAL_CONTROLLER_UID,
        validator_uid="round12_typed_evidence_relationship_validator",
        validator_report={
            "validator_uid": "round12_typed_evidence_relationship_validator",
            "status": "passed",
            "assessment": gate_status["status"],
        },
        input_artifact_hashes={
            "execution_input_bundle": observed_bundle_sha256,
            "relationship_graph": sha256_payload(graph),
            "frozen_gap_source": sha256_file(frozen_validation_path),
        },
        config_hash=sha256_file(repository_root / "Libs/configs/schema/schemas.yaml"),
        commit=git_commit(repository_root),
        decision="blocked",
        blockers=gate_blockers,
        lineage_parent="round_11g_targeted_repair_and_full_suite_closure",
    ) | {
        "round_uid": "round_12_typed_evidence_relationships",
        **gate_status,
        "validation_status": validation["status"],
        "execution_identity": {
            "code_commit": expected_code_commit,
            "dirty_status": execution_dirty_status,
            "code_relation": "not_exact_clean_commit" if execution_dirty_status != "clean" else "exact_clean_commit",
            "input_relation": "exact_frozen_manifest",
            "execution_input_bundle_sha256": observed_bundle_sha256,
        },
        "frozen_gap_count": len(gaps),
        "blockers": gate_blockers,
        "blocker_count": len(gate_blockers),
        "evidence_fabric": {"sealed": False},
        "dt_core": {"status": "not_evaluated"},
        "round_11_closure": "accepted_unchanged",
    }
    return {"graph": graph, "gaps": gaps, "validation": validation, "gate": gate}


def _json_cell(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_round12_artifacts(payloads: Mapping[str, object], repository_root: Path = REPO_ROOT) -> None:
    graph = dict(payloads["graph"])
    write_json(repository_root / OUTPUT_PATHS["graph"], graph)
    write_json(repository_root / OUTPUT_PATHS["validation"], payloads["validation"])
    write_json(repository_root / OUTPUT_PATHS["gate"], payloads["gate"])
    write_tsv(repository_root / OUTPUT_PATHS["nodes"], ["schema_version", "node_uid", "node_type", "evidence_class", "qc_grade", "identity_axes", "provenance_ref", "train_eval_role", "status"], [{**row, "identity_axes": _json_cell(row["identity_axes"])} for row in graph["nodes"]])
    write_tsv(repository_root / OUTPUT_PATHS["relations"], ["schema_version", "relation_uid", "relation_type", "source_node_uid", "target_node_uid", "directionality", "biological_unit_relation", "temporal_relation", "action_relation", "context_relation", "support_domain_ref", "eligibility_ref", "provenance_ref", "confidence", "status"], graph["relations"])
    write_tsv(repository_root / OUTPUT_PATHS["pairings"], ["schema_version", "pairing_uid", "source_node_uid", "target_node_uid", "decision", "identity_axis_results", "required_identity_axes", "pairing_evidence", "blockers", "status"], [{**row, "identity_axis_results": _json_cell(row["identity_axis_results"]), "required_identity_axes": _json_cell(row["required_identity_axes"]), "pairing_evidence": _json_cell(row["pairing_evidence"]), "blockers": _json_cell(row["blockers"])} for row in graph["pairing_decisions"]])
    write_tsv(repository_root / OUTPUT_PATHS["domains"], ["schema_version", "support_domain_uid", *SUPPORT_DOMAIN_FIELDS, "generalization_status", "provenance_ref", "status"], [{**row, **{field: _json_cell(row[field]) for field in SUPPORT_DOMAIN_FIELDS}} for row in graph["support_domains"]])
    write_tsv(repository_root / OUTPUT_PATHS["eligibilities"], list(graph["eligibilities"][0]), [{**row, "reasons": _json_cell(row["reasons"])} for row in graph["eligibilities"]])
    write_tsv(repository_root / OUTPUT_PATHS["conflicts"], ["schema_version", "conflict_uid", "conflict_type", "member_uids", "state", "typed_context_difference", "provenance_ref", "status"], [{**row, "member_uids": _json_cell(row["member_uids"]), "typed_context_difference": _json_cell(row["typed_context_difference"])} for row in graph["conflicts"]])
    write_tsv(repository_root / OUTPUT_PATHS["gaps"], ["gap_uid", "missing_node_type", "missing_relationship_type", "required_identity_axes", "available_evidence", "blocked_pairing_reason", "affected_downstream_role", "resolution_owner", "status"], list(payloads["gaps"]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Round-12 typed evidence relationship contracts and pilot catalogs.")
    parser.add_argument("--execution-input-bundle", required=True)
    parser.add_argument("--expected-bundle-sha256", required=True)
    parser.add_argument("--frozen-validation", default="Output/audits/dt_evidence_fabric_validation.json")
    parser.add_argument("--expected-code-commit", required=True)
    parser.add_argument("--execution-dirty-status", choices=("clean", "dirty"), default="dirty")
    args = parser.parse_args()
    payloads = build_round12_artifacts(
        execution_bundle_path=Path(args.execution_input_bundle),
        expected_bundle_sha256=args.expected_bundle_sha256,
        frozen_validation_path=REPO_ROOT / args.frozen_validation,
        expected_code_commit=args.expected_code_commit,
        execution_dirty_status=args.execution_dirty_status,
    )
    write_round12_artifacts(payloads)
    print(f"{payloads['gate']['status']} validation={payloads['validation']['status']} gaps={len(payloads['gaps'])}")
    return 0 if payloads["validation"]["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
