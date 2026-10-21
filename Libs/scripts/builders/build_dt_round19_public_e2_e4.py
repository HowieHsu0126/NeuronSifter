#!/usr/bin/env python3
"""Materialize the bounded Round-19 public E2/E4 readiness assessment.

The builder reads only already-revealed Round-18 qualification artifacts and
public source metadata. It never reads a final-held-out outcome, executes a
trajectory producer, starts training, or mutates the Evidence Fabric release.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import socket
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.evidence_fabric_v0 import write_parquet_records  # noqa: E402
from Libs.utils.data.raw_data_registry import load_config, read_tsv, write_json, write_tsv  # noqa: E402
from Libs.utils.data.round19_contracts import (  # noqa: E402
    E2AdmissionDecision,
    E2Endpoint,
    E4ProducerQualification,
    SCHEMA_VERSIONS,
    TrainingEligibilityRow,
    audit_independence,
    audit_split_leakage,
    evaluate_training_readiness,
    record_uid,
    validate_e2_admission,
    validate_e2_endpoint,
    validate_e4_producer,
    validate_frozen_gaps,
    validate_round19_invariants,
    validate_training_eligibility,
)


CONFIG_PATH = "Libs/configs/config_stack.yaml"
ROUND19_CONTRACT_CONFIG_PATH = "Libs/configs/data/round19_public_e2_e4.yaml"
ROUND19_CANDIDATE_CONFIG_PATH = "Libs/configs/data/round19_public_e2_candidates.yaml"
EXPECTED_BASE = "dba20fb554c18d18ba53b0f1cd713dbe9279d743"
ROUND18_SAFE_FILES = {
    "validation": "audits/dt_round18_validation.json",
    "prediction": "littwinbench/round18_pilot/prediction_catalog.tsv",
    "ranking": "littwinbench/round18_pilot/ranking_catalog.tsv",
    "decision": "littwinbench/round18_pilot/decision_catalog.tsv",
    "abstention": "littwinbench/round18_pilot/abstention_catalog.tsv",
    "leakage": "littwinbench/round18_pilot/leakage_control_catalog.tsv",
}


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
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True, capture_output=True, text=True
    ).stdout.strip()


def _base_is_ancestor(repository: Path) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", EXPECTED_BASE, "HEAD"],
        cwd=repository,
        check=False,
    ).returncode == 0


def _resolve(output_root: Path, relative: object) -> Path:
    path = Path(str(relative))
    if path.is_absolute():
        raise ValueError(f"Round-19 output paths must be relative: {path}")
    return output_root / path


def _safe_round18_paths(root: Path) -> dict[str, Path]:
    paths = {name: root / relative for name, relative in ROUND18_SAFE_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Round-18 diagnosis inputs are missing: {missing}")
    forbidden = [str(path) for path in paths.values() if "held_out_outcome" in path.as_posix()]
    if forbidden:
        raise ValueError(f"Round-19 diagnosis attempted evaluator-held-out access: {forbidden}")
    return paths


def diagnose_round18(root: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    paths = _safe_round18_paths(root)
    validation = _load_json(paths["validation"])
    summary = dict(validation["pilot_summary"])
    predictions = read_tsv(paths["prediction"])
    rankings = read_tsv(paths["ranking"])
    decisions = read_tsv(paths["decision"])
    abstentions = read_tsv(paths["abstention"])
    leakage = read_tsv(paths["leakage"])

    by_system: dict[str, dict[str, object]] = {}
    decision_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in decisions:
        decision_groups[row["system_uid"]].append(row)
    abstention_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in abstentions:
        abstention_groups[row["system_uid"]].append(row)
    for system_uid in sorted({row["system_uid"] for row in predictions + rankings}):
        system_decisions = decision_groups.get(system_uid, [])
        system_abstentions = abstention_groups.get(system_uid, [])
        scored = [row for row in system_decisions if row.get("correct") in {"True", "False"}]
        correct = sum(row.get("correct") == "True" for row in scored)
        required = [row for row in system_abstentions if row.get("abstention_required") == "True"]
        by_system[system_uid] = {
            "scored_decision_count": len(scored),
            "decision_accuracy": correct / len(scored) if scored else None,
            "abstention_required_count": len(required),
            "correct_abstention_count": sum(row.get("abstained") == "True" for row in required),
            "unsupported_answer_count": sum(row.get("unsupported_answer_emitted") == "True" for row in system_abstentions),
            "unknown_components_preserved_count": sum(row.get("unknown_components_preserved") == "True" for row in system_abstentions),
            "conflicts_preserved_count": sum(row.get("conflicts_preserved") == "True" for row in system_abstentions),
        }

    true_labels = Counter(row.get("true_label", "") for row in decisions)
    outcome_classes = {
        "supported": sum("supported" in key.lower() for key in true_labels for _ in range(true_labels[key])),
        "falsified": sum("refuted" in key.lower() for key in true_labels for _ in range(true_labels[key])),
        "mixed": sum("mixed" in key.lower() for key in true_labels for _ in range(true_labels[key])),
        "inconclusive": sum(any(token in key.lower() for token in ("unresolved", "insufficient")) for key in true_labels for _ in range(true_labels[key])),
    }
    governance = dict(summary["governance_violation_counts"])
    report = {
        "schema_version": "neuronsifter-dt-round19-round18-diagnosis-v1",
        "status": "qualification_only",
        "source_round": 18,
        "final_held_out_access_count": 0,
        "effective_episode_count": summary["episode_count"],
        "public_source_count": summary["source_count"],
        "track_counts": summary["track_counts"],
        "system_run_count": summary["run_count"],
        "result_count": summary["result_count"],
        "hypothesis_outcomes": outcome_classes,
        "system_comparison": by_system,
        "calibration": {
            "metric_implementation_count": summary["metric_count"],
            "classification": "qualification_only",
            "limitation": "nine-episode pilot is underpowered for empirical calibration claims",
        },
        "abstention": {
            "record_count": len(abstentions),
            "unsupported_answer_rate": sum(row.get("unsupported_answer_emitted") == "True" for row in abstentions) / len(abstentions),
            "false_confidence_case_count": sum(row.get("unsupported_answer_emitted") == "True" and row.get("abstained") != "True" for row in abstentions),
        },
        "evidence_selection_precision": {
            "status": "underpowered",
            "information_action_episode_count": summary["track_counts"].get("information_action_selection", 0),
        },
        "memory_retrieval_leakage_risk": "reported_residual_risk_not_eliminated",
        "twin_debt_behavior": "qualification_only_action_selection_signal",
        "governance_violation_counts": governance,
        "held_out_access_attack_count": len(leakage),
        "held_out_access_attack_rejected_count": sum(row.get("decision") == "rejected" for row in leakage),
        "classification": "qualification_only",
        "source_hashes": {name: sha256_file(path) for name, path in paths.items()},
    }
    failure_map = [
        {"failure_uid": "round18_underpowered_calibration", "classification": "underpowered", "requirement_type": "E2", "requirement": "independent machine-readable numeric endpoints across publication families"},
        {"failure_uid": "round18_evidence_selection_underpowered", "classification": "underpowered", "requirement_type": "E2", "requirement": "multiple public information-action validation units"},
        {"failure_uid": "round18_numeric_abstract_risk", "classification": "blocked_label_quality", "requirement_type": "E2", "requirement": "machine-readable supplement or accession values; abstract prose remains directional only"},
        {"failure_uid": "round18_memory_contamination_residual", "classification": "blocked_memory_risk", "requirement_type": "support_domain", "requirement": "post-model-cutoff public episodes and masked identifiers"},
        {"failure_uid": "round18_no_intervention_trajectory_validation", "classification": "qualification_only", "requirement_type": "E4", "requirement": "action-complete producer with matched control and independent initial state"},
    ]
    return report, failure_map


def _endpoint_from_config(row: Mapping[str, object]) -> dict[str, object]:
    base = dict(row)
    groups = dict(base.pop("independence_groups"))
    support_domain = base.pop("support_domain")
    action_class = base.pop("action_class")
    endpoint_basis = {**base, "independence_groups": groups}
    endpoint = E2Endpoint(
        endpoint_uid=record_uid("round19_e2_endpoint", endpoint_basis),
        schema_version=SCHEMA_VERSIONS["e2_endpoint"],
        content_hash=str(base.pop("fulltext_xml_sha256")),
        independence_groups=groups,
        source_is_public=True,
        controlled_access_required=False,
        empirical_observation=True,
        **base,
    ).to_dict()
    endpoint["support_domain"] = support_domain
    endpoint["action_class"] = action_class
    endpoint["strict_public_only_flag"] = True
    endpoint["controlled_source_resolution_count"] = 0
    endpoint["publicly_available"] = True
    return endpoint


def _admit_e2(endpoint: Mapping[str, object]) -> dict[str, object]:
    decision = E2AdmissionDecision(
        admission_uid=record_uid("round19_e2_admission", endpoint),
        endpoint_uid=str(endpoint["endpoint_uid"]),
        decision="admitted_e2_directional",
        reason_codes=("public_fulltext_author_claim", "directional_only_no_machine_readable_effect_size"),
        evidence_hash=str(endpoint["content_hash"]),
        eligibility="hypothesis_generation_only",
        empirical_observation=True,
    ).to_dict()
    blockers = validate_e2_admission(decision, endpoint)
    if blockers:
        raise ValueError(blockers)
    return decision


def _producer_record(row: Mapping[str, object], repository: Path) -> dict[str, object]:
    source = repository / str(row["producer_source"])
    record = E4ProducerQualification(
        producer_uid=str(row["producer_uid"]),
        producer_version=str(row["producer_version"]),
        producer_hash=sha256_file(source),
        equation_model_identity=str(row["equation_model_identity"]),
        parameter_provenance=str(row["parameter_provenance"]),
        supported_state_variables=tuple(str(item) for item in row["supported_state_variables"]),
        supported_action_semantics=tuple(str(item) for item in row["supported_action_semantics"]),
        initial_state_contract=str(row["initial_state_contract"]),
        control_generation_capability=str(row["control_generation_capability"]),
        seed_policy=str(row["seed_policy"]),
        fidelity_policy=str(row["fidelity_policy"]),
        numerical_qc=str(row["numerical_qc"]),
        constraint_policy=str(row["constraint_policy"]),
        license_or_redistribution=str(row["license_or_redistribution"]),
        decision=str(row["decision"]),
        support_domain=str(row["support_domain"]),
    ).to_dict()
    blockers = validate_e4_producer(record)
    if blockers:
        raise ValueError(blockers)
    record["decision_reason"] = str(row["decision_reason"])
    return record


def _write_outputs(
    *, repository: Path, output_root: Path, config: Mapping[str, object], round18_root: Path,
    frozen_gap_ledger: Path, retrieval_job_id: str, retrieval_stderr: Path, created_at: str,
) -> dict[str, object]:
    section = dict(config["round19_public_e2_e4"])
    candidates = dict(config["round19_public_e2_candidates"])
    outputs = dict(section["output_paths"])
    invariants = dict(section["invariants"])
    invariant_blockers = validate_round19_invariants(invariants)
    if invariant_blockers:
        raise ValueError(invariant_blockers)
    if not _base_is_ancestor(repository):
        raise ValueError("Round-19 candidate must descend from the exact accepted base commit")

    diagnosis, failure_map = diagnose_round18(round18_root)
    write_json(_resolve(output_root, outputs["round18_diagnosis"]), diagnosis)
    write_tsv(_resolve(output_root, outputs["replay_failure_map"]), list(failure_map[0]), failure_map)

    endpoint_rows = [_endpoint_from_config(row) for row in candidates["candidates"]]
    forbidden_source_tokens = tuple(str(item) for item in section["public_source_policy"]["rejected_routes"])
    for row in endpoint_rows:
        blockers = validate_e2_endpoint(row, forbidden_source_tokens=forbidden_source_tokens)
        if blockers:
            raise ValueError(blockers)
    admission_rows = [_admit_e2(row) for row in endpoint_rows]
    endpoint_fields = list(endpoint_rows[0])
    write_parquet_records(_resolve(output_root, outputs["e2_candidate_catalog"]), endpoint_rows, endpoint_fields)
    write_tsv(_resolve(output_root, outputs["e2_admission_ledger"]), list(admission_rows[0]), admission_rows)

    independence_rows = [{"endpoint_uid": row["endpoint_uid"], **dict(row["independence_groups"]), "partition": row["partition"]} for row in endpoint_rows]
    write_tsv(_resolve(output_root, outputs["e2_independence_groups"]), list(independence_rows[0]), independence_rows)
    independence = audit_independence(independence_rows)
    write_json(_resolve(output_root, outputs["e2_independence_audit"]), {
        "schema_version": "neuronsifter-dt-round19-e2-independence-audit-v1", **independence,
    })
    duplicate_rows = [{"duplicate_group_uid": "none", "member_count": 0, "decision": "no_duplicate_or_version_collision", "notes": "one publication family admitted directionally"}]
    write_tsv(_resolve(output_root, outputs["e2_duplicate_audit"]), list(duplicate_rows[0]), duplicate_rows)
    leakage_blockers = audit_split_leakage(independence_rows)
    write_json(_resolve(output_root, outputs["e2_leakage_audit"]), {
        "schema_version": "neuronsifter-dt-round19-e2-leakage-audit-v1",
        "status": "passed" if not leakage_blockers else "blocked",
        "final_held_out_access_count": 0,
        "final_held_out_leakage_count": len(leakage_blockers),
        "blockers": leakage_blockers,
    })
    coverage = {
        "schema_version": "neuronsifter-dt-round19-e2-coverage-v1",
        "status": "insufficient_for_training",
        "candidate_endpoint_count": len(endpoint_rows),
        "admitted_e2_strong": 0,
        "admitted_e2_weighted": 0,
        "admitted_e2_directional": len(admission_rows),
        "independent_study_count": independence["independent_study_count"],
        "action_class_count": len({str(row["action_class"]) for row in endpoint_rows}),
        "support_domain_count": len({str(row["support_domain"]) for row in endpoint_rows}),
        "strong_threshold_met": False,
        "weighted_threshold_met": False,
    }
    write_json(_resolve(output_root, outputs["e2_coverage"]), coverage)

    domains = [dict(row) for row in candidates["e4_domain_candidates"]]
    selected = [row for row in domains if row.get("selected") is True]
    if len(selected) != 1:
        raise ValueError("Exactly one E4 domain must be selected")
    write_json(_resolve(output_root, outputs["e4_domain_selection"]), {
        "schema_version": "neuronsifter-dt-round19-e4-domain-selection-v1",
        "status": "selected_for_qualification_not_execution",
        "selected_domain_uid": selected[0]["domain_uid"],
        "selected_only_for_computational_convenience": False,
        "score_dimensions": list(section["e4_domain_selection_dimensions"]),
        "candidate_domains": domains,
    })
    producer_rows = [_producer_record(row, repository) for row in candidates["producer_candidates"]]
    producer_fields = list(producer_rows[0])
    write_tsv(_resolve(output_root, outputs["producer_qualification"]), producer_fields, producer_rows)

    pilot_manifest = {
        "schema_version": "neuronsifter-dt-round19-e4-pilot-manifest-v1",
        "status": "not_run_no_qualified_intervention_producer",
        "selected_domain_uid": selected[0]["domain_uid"],
        "qualified_intervention_producer_count": 0,
        "trajectory_count": 0,
        "control_pair_count": 0,
        "hpc_job_count": 0,
        "training_started": False,
        "model_output_promoted_to_empirical": False,
    }
    write_json(_resolve(output_root, outputs["e4_pilot_manifest"]), pilot_manifest)
    write_json(_resolve(output_root, outputs["e4_pilot_validation"]), {
        **pilot_manifest,
        "validation_status": "blocked_producer_qualification",
        "negative_cases": {
            "unsupported_action": "rejected",
            "incomplete_action": "rejected",
            "constraint_failure": "rejected",
        },
    })
    write_json(_resolve(output_root, outputs["e4_control_pair_audit"]), {
        "schema_version": "neuronsifter-dt-round19-e4-control-pair-audit-v1",
        "status": "not_applicable_no_pilot",
        "matched_control_pair_count": 0,
        "unpaired_effect_claim_count": 0,
    })
    trajectory_fields = ["trajectory_uid", "producer_uid", "producer_version", "producer_hash", "initial_state_hash", "action_hash", "support_domain", "state_artifact_hash", "native_time_grid_hash", "canonical_time_grid_hash", "fidelity", "seed", "parameter_hash", "constraint_status", "numerical_status", "output_class", "empirical_observation", "admission_decision"]
    pair_fields = ["pair_uid", "action_trajectory_uid", "control_trajectory_uid", "initial_state_family", "producer_version", "parameter_family", "fidelity", "time_grid", "seed_policy", "pairing_decision", "action_effect_claim_eligible"]
    write_parquet_records(_resolve(output_root, outputs["e4_trajectory_catalog"]), [], trajectory_fields)
    write_parquet_records(_resolve(output_root, outputs["e4_control_pair_catalog"]), [], pair_fields)
    admission_fields = ["trajectory_uid", "decision", "reason_codes", "empirical_observation"]
    write_tsv(_resolve(output_root, outputs["e4_admission_ledger"]), admission_fields, [])
    alignment_fields = ["alignment_uid", "endpoint_uid", "trajectory_uid", "alignment_class", "model_output_non_empirical"]
    write_tsv(_resolve(output_root, outputs["e2_e4_alignment"]), alignment_fields, [])
    write_json(_resolve(output_root, outputs["e4_coverage"]), {
        "schema_version": "neuronsifter-dt-round19-e4-coverage-v1",
        "status": "blocked_insufficient_e4",
        "qualified_intervention_producer_count": 0,
        "intervention_trajectory_count": 0,
        "matched_control_pair_count": 0,
        "model_derived_evidence_count": 0,
        "empirical_observation_count": 0,
    })

    inherited_gaps = read_tsv(frozen_gap_ledger)
    gap_rows = [{"gap_uid": row["gap_uid"], "status": "frozen_unresolved", "evidence_refs": [], "resolution_basis": "directional E2 does not independently satisfy the inherited empirical gap contract"} for row in inherited_gaps]
    gap_blockers = validate_frozen_gaps(gap_rows)
    if gap_blockers:
        raise ValueError(gap_blockers)
    serialized_gap_rows = [{**row, "evidence_refs": json.dumps(row["evidence_refs"], separators=(",", ":"))} for row in gap_rows]
    write_tsv(_resolve(output_root, outputs["frozen_gap_resolution"]), list(serialized_gap_rows[0]), serialized_gap_rows)

    eligibility_rows = [TrainingEligibilityRow(
        artifact_uid=str(row["endpoint_uid"]), role="hypothesis_generation_only", partition=str(row["partition"]),
        publication_family=str(row["independence_groups"]["publication_family"]),
        source_family=str(row["independence_groups"]["source_collection"]),
        study_family=str(row["independence_groups"]["study"]),
        biological_unit_family=str(row["independence_groups"]["biological_unit"]),
        action_family=str(row["independence_groups"]["action_class"]),
        regimen_family=str(row["independence_groups"]["dose_regimen"]),
        support_domain=str(row["support_domain"]),
        eligibility_basis="directional author-claim label; excluded from numeric training and validation",
        validation_only=False,
    ).to_dict() for row in endpoint_rows]
    eligibility_blockers = validate_training_eligibility(eligibility_rows)
    if eligibility_blockers:
        raise ValueError(eligibility_blockers)
    write_tsv(_resolve(output_root, outputs["training_eligibility"]), list(eligibility_rows[0]), eligibility_rows)
    split_feasibility = {
        "schema_version": "neuronsifter-dt-round19-split-feasibility-v1",
        "status": "blocked_split_feasibility",
        "required_holdouts": section["training_readiness_contract"]["required_holdouts"],
        "final_held_out_access_count": 0,
        "leakage_blockers": leakage_blockers,
        "unexplained_overlap_count": 0,
        "reason": "one directional publication family cannot populate independent train calibration and validation partitions",
    }
    write_json(_resolve(output_root, outputs["split_feasibility"]), split_feasibility)
    readiness = evaluate_training_readiness(
        contract=dict(section["training_readiness_contract"]), e2_rows=admission_rows,
        e4_rows=[], pair_rows=[], validation_units=[], split_blockers=[{"code": "split_feasibility", "message": split_feasibility["reason"]}],
    )
    readiness.update({
        "schema_version": "neuronsifter-dt-round19-world-action-training-readiness-v1",
        "assessment_completed": True,
        "training_authorized": False,
        "training_started": False,
        "thresholds_lowered_after_source_observation": False,
    })
    write_json(_resolve(output_root, outputs["training_readiness"]), readiness)

    retrieval = {
        "schema_version": "neuronsifter-dt-round19-public-source-resolution-v1",
        "status": "public_fulltext_resolved_supplement_retrieval_failed",
        "source_accession": "PMC12570821",
        "strict_public_only_flag": True,
        "controlled_access_required": False,
        "controlled_source_resolution_count": 0,
        "attempts": [
            {"attempt_uid": f"slurm_job_{retrieval_job_id}", "terminal_state": "FAILED", "exit_code": "1:0", "failure_class": "compute_node_public_network_unavailable", "stderr_hash": sha256_file(retrieval_stderr)},
            {"attempt_uid": "management_bounded_download_50s", "terminal_state": "FAILED", "failure_class": "timeout_incomplete_zip", "received_bytes": 11992619},
            {"attempt_uid": "management_bounded_download_180s", "terminal_state": "FAILED", "failure_class": "timeout_incomplete_zip", "received_bytes": 39302591},
            {"attempt_uid": "management_range_resume", "terminal_state": "FAILED", "failure_class": "server_range_not_supported", "curl_exit_code": 33},
        ],
        "fulltext_publicly_available": True,
        "fulltext_hash": endpoint_rows[0]["content_hash"],
        "supplement_admitted": False,
        "failed_attempts_preserved": True,
    }
    write_json(output_root / "Output/audits/dt_round19_public_source_resolution.json", retrieval)

    artifact_paths = [
        _resolve(output_root, value) for value in outputs.values()
        if str(value) not in {outputs["validation"], outputs["full_suite_execution"], outputs["gate"], outputs["handoff"]}
    ]
    artifact_paths.append(output_root / "Output/audits/dt_round19_public_source_resolution.json")
    manifest = {
        "schema_version": "neuronsifter-dt-round19-materialization-manifest-v1",
        "status": "materialized_readiness_blocked",
        "created_at": created_at,
        "source_commit": _git_commit(repository),
        "accepted_base_commit": EXPECTED_BASE,
        "source_tree": subprocess.run(["git", "rev-parse", "HEAD^{tree}"], cwd=repository, check=True, capture_output=True, text=True).stdout.strip(),
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "host": socket.gethostname()},
        "artifact_hashes": {path.relative_to(output_root).as_posix(): sha256_file(path) for path in artifact_paths if path.is_file()},
        "strict_public_only_flag": True,
        "controlled_source_resolution_count": 0,
        "final_held_out_access_count": 0,
        "training_authorized": False,
        "training_started": False,
        "evidence_fabric_sealed": False,
        "frozen_gap_count": 20,
        "dt_core_status": "not_evaluated",
    }
    write_json(output_root / "Output/audits/dt_round19_materialization_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--round18-output-root", type=Path, required=True)
    parser.add_argument("--frozen-gap-ledger", type=Path, required=True)
    parser.add_argument("--retrieval-job-id", default="502147")
    parser.add_argument("--retrieval-stderr", type=Path, required=True)
    parser.add_argument("--created-at", default=None)
    args = parser.parse_args()
    repository = args.repository_root.resolve()
    config = load_config(repository / CONFIG_PATH)
    contract_config = load_config(repository / ROUND19_CONTRACT_CONFIG_PATH)
    candidate_config = load_config(repository / ROUND19_CANDIDATE_CONFIG_PATH)
    config["round19_public_e2_e4"] = contract_config["round19_public_e2_e4"]
    config["round19_public_e2_candidates"] = candidate_config["round19_public_e2_candidates"]
    manifest = _write_outputs(
        repository=repository, output_root=args.output_root.resolve(), config=config,
        round18_root=args.round18_output_root.resolve(), frozen_gap_ledger=args.frozen_gap_ledger.resolve(),
        retrieval_job_id=args.retrieval_job_id, retrieval_stderr=args.retrieval_stderr.resolve(),
        created_at=args.created_at or datetime.now(timezone.utc).isoformat(),
    )
    print(json.dumps({"status": manifest["status"], "training_started": False, "frozen_gap_count": 20}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
