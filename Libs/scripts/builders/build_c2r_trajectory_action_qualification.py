"""Qualify a versioned C2 trajectory-action registry without mutating prior evidence.

This builder is deliberately fail-closed.  It audits existing data and may
describe a candidate ``trajectory_action_registry_v2``, but it never copies,
repairs, or re-labels legacy C2 trajectory rows.  A registry is sealed only
when every entry-gate prerequisite is satisfied.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from collections import Counter
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.public_foundation_corpus import canonical_sha256  # noqa: E402
from Libs.utils.data.public_foundation_evidence import sha256_file  # noqa: E402
from Libs.utils.data.raw_data_registry import load_config, read_tsv, write_json, write_tsv  # noqa: E402


MATRIX_FIELDS = [
    "trajectory_family_uid", "simulator_family_uid", "action_family_uid", "trajectory_count",
    "qc_grade", "source", "publication_model_provenance", "generator_implementation",
    "equation_identity", "parameter_lineage", "action_uid", "dose_value", "dose_unit",
    "duration", "schedule", "route", "exposure_program_ref", "occupancy_program_ref",
    "initial_state_identity", "time_grid", "region_axis", "compartment_axis", "species_axis",
    "independence_group", "independence_level", "eligible_train_eval_role",
]
QUARANTINE_FIELDS = [
    "action_uid", "candidate_domain", "intervention_type", "agent_set", "target_set", "dose_value",
    "dose_unit", "duration", "schedule", "route", "exposure_program_ref", "occupancy_program_ref",
    "source_provenance", "missing_fields", "missing_reason", "role", "headline_eligible",
]
REAL_CANDIDATE_FIELDS = [
    "candidate_uid", "source", "publication_accession", "block_hash", "qc_grade", "observation_count",
    "effective_sample_size", "baseline_observation", "structured_action", "post_action_observation",
    "time_or_ordered_state", "biological_unit_linkage", "trajectory_status", "decision", "reason",
]
OPTION_FIELDS = [
    "option_id", "option_name", "trajectory_families", "simulator_families", "independence_status",
    "trajectory_count", "action_completeness", "real_observation_linked_count", "holdouts_enabled",
    "baseline_compatibility", "storage", "compute", "implementation_cost", "scientific_claim_boundary",
    "strict_action_holdout", "strict_trajectory_or_simulator_holdout", "fair_action_ablation",
    "deterministic_stochastic_shared_domain", "bootstrap_pairing_unit", "status", "selected",
]


def _repo_path(repo_root: Path, relative: str) -> Path:
    path = (repo_root / relative).resolve()
    root = repo_root.resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"C2R path escapes repository: {relative}")
    return path


def _json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _binding(repo_root: Path, relative: str) -> dict[str, object]:
    path = _repo_path(repo_root, relative)
    return {
        "path": relative,
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def _value(value: object, unavailable: str = "unavailable") -> str:
    text = str(value or "").strip()
    return text if text else unavailable


def _load_required_json(repo_root: Path, inputs: Mapping[str, object]) -> dict[str, dict[str, object]]:
    payloads: dict[str, dict[str, object]] = {}
    for key, relative in inputs.items():
        if not key.startswith("c2_"):
            continue
        path = _repo_path(repo_root, str(relative))
        payloads[key] = _json(path)
    return payloads


def _immutable_bindings(repo_root: Path, entries: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    bindings = []
    for entry in entries:
        path = _repo_path(repo_root, str(entry["evidence_bundle_path"]))
        payload = _json(path)
        # V1 sealed the canonical value as corpus_hash; V2 retained the same
        # immutable value under the explicit corpus_identity_hash field.
        observed = str(payload.get("corpus_identity_hash") or payload.get("corpus_hash") or "")
        expected = str(entry["identity_hash"])
        bindings.append({
            "corpus_uid": entry["corpus_uid"],
            "expected_identity_hash": expected,
            "observed_identity_hash": observed or None,
            "identity_match": observed == expected,
            "evidence_bundle": _binding(repo_root, str(entry["evidence_bundle_path"])),
        })
    return bindings


def _eligible_legacy_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            row for row in csv.DictReader(handle, delimiter="\t")
            if row.get("qc_grade") in {"A", "B"} and row.get("qc_use") == "main_pretraining"
        ]


def _legacy_matrix_rows(
    rows: Iterable[Mapping[str, str]],
    regimen_by_uid: Mapping[str, Mapping[str, str]],
    domains: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in rows:
        source_collection = str(row["source_collection"])
        domain_key = "legacy_full_microenvironment" if source_collection == "full_microenvironment_teacher" else "legacy_mesoscopic"
        domain = domains[domain_key]
        action = regimen_by_uid.get(str(row["regimen_uid"]), {})
        output.append({
            "trajectory_family_uid": row["trajectory_family_uid"],
            "simulator_family_uid": domain["simulator_family_uid"],
            "action_family_uid": row["regimen_uid"],
            "trajectory_count": 1,
            "qc_grade": row["qc_grade"],
            "source": row["source_collection"],
            "publication_model_provenance": "repository-generated C2 legacy simulator; no publication-bound action execution record",
            "generator_implementation": row["teacher_family"],
            "equation_identity": domain["equation_identity"],
            "parameter_lineage": f"{domain['parameter_lineage']}:{row['parameter_profile_uid']}",
            "action_uid": row["regimen_uid"],
            "dose_value": "unavailable",
            "dose_unit": "unavailable",
            "duration": "unavailable",
            "schedule": _value(action.get("schedule") or row.get("schedule")),
            "route": _value(action.get("route") or row.get("route")),
            "exposure_program_ref": "unavailable",
            "occupancy_program_ref": "unavailable",
            "initial_state_identity": row["paired_uid"],
            "time_grid": "28",
            "region_axis": "6",
            "compartment_axis": "6",
            "species_axis": "15",
            "independence_group": row["paired_uid"],
            "independence_level": domain["independence_level"],
            "eligible_train_eval_role": "quarantined_missing_physical_action_provenance",
        })
    return output


def _osp_matrix_rows(rows: Iterable[Mapping[str, str]], domain: Mapping[str, object]) -> list[dict[str, object]]:
    output = []
    for row in rows:
        output.append({
            "trajectory_family_uid": row["pbpk_rollout_uid"],
            "simulator_family_uid": domain["simulator_family_uid"],
            "action_family_uid": row["regimen_uid"],
            "trajectory_count": 1,
            "qc_grade": "unqualified_for_c2_registry",
            "source": row["source_repository"],
            "publication_model_provenance": row["source_model_id"],
            "generator_implementation": "Libs/scripts/teacher_generation/build_osp_pbpk_teacher_rollouts.py",
            "equation_identity": domain["equation_identity"],
            "parameter_lineage": domain["parameter_lineage"],
            "action_uid": row["regimen_uid"],
            "dose_value": "unavailable",
            "dose_unit": "unavailable",
            "duration": "unavailable",
            "schedule": _value(row.get("schedule")),
            "route": _value(row.get("route")),
            "exposure_program_ref": "unavailable",
            "occupancy_program_ref": "unavailable",
            "initial_state_identity": "unavailable",
            "time_grid": "28",
            "region_axis": "6",
            "compartment_axis": "6",
            "species_axis": "15",
            "independence_group": row["source_model_id"],
            "independence_level": domain["independence_level"],
            "eligible_train_eval_role": "quarantined_missing_physical_action_provenance",
        })
    return output


def _neuron_matrix_rows(rows: Iterable[Mapping[str, str]], domain: Mapping[str, object]) -> list[dict[str, object]]:
    output = []
    for row in rows:
        output.append({
            "trajectory_family_uid": row["neuron_rxd_protocol_uid"],
            "simulator_family_uid": domain["simulator_family_uid"],
            "action_family_uid": row["neuron_rxd_protocol_uid"],
            "trajectory_count": 1,
            "qc_grade": "unqualified_for_c2_registry",
            "source": row["source_repository"],
            "publication_model_provenance": row["source_model_id"],
            "generator_implementation": "Libs/scripts/teacher_generation/run_neuron_rxd_microenvironment_teacher.py",
            "equation_identity": domain["equation_identity"],
            "parameter_lineage": domain["parameter_lineage"],
            "action_uid": row["neuron_rxd_protocol_uid"],
            "dose_value": "0.15",
            "dose_unit": "nA",
            "duration": "120 ms",
            "schedule": "single_current_clamp",
            "route": "not_applicable",
            "exposure_program_ref": "unavailable:not_a_drug_exposure",
            "occupancy_program_ref": "unavailable:not_a_drug_action",
            "initial_state_identity": "resting_voltage_-65mV",
            "time_grid": "28",
            "region_axis": "6",
            "compartment_axis": "6",
            "species_axis": "15",
            "independence_group": row["source_model_id"],
            "independence_level": domain["independence_level"],
            "eligible_train_eval_role": "dynamics_only_single_protocol_not_action_holdout_eligible",
        })
    return output


def _legacy_action_quarantine(rows: Iterable[Mapping[str, str]]) -> list[dict[str, object]]:
    missing = [
        "dose_value", "dose_unit", "duration", "exposure_program_ref", "occupancy_program_ref",
        "publication_accession", "assay_series_uid",
    ]
    return [{
        "action_uid": row["regimen_uid"],
        "candidate_domain": "legacy_c2_simulator",
        "intervention_type": "compound_regimen",
        "agent_set": _value(row.get("compound_uid_set")),
        "target_set": _value(row.get("target_uid_set")),
        "dose_value": "unavailable",
        "dose_unit": "unavailable",
        "duration": "unavailable",
        "schedule": _value(row.get("schedule")),
        "route": _value(row.get("route")),
        "exposure_program_ref": "unavailable",
        "occupancy_program_ref": "unavailable",
        "source_provenance": "action_ready synthetic dose multiplier registry",
        "missing_fields": "|".join(missing),
        "missing_reason": "Only dose_level_id and dose_multiplier are recorded; physical values and execution provenance were not observed.",
        "role": "quarantine",
        "headline_eligible": False,
    } for row in rows]


def _osp_action_quarantine(rows: Iterable[Mapping[str, str]]) -> list[dict[str, object]]:
    return [{
        "action_uid": row["regimen_uid"],
        "candidate_domain": "osp_pbpk",
        "intervention_type": "compound_regimen",
        "agent_set": _value(row.get("canonical_compound_id")),
        "target_set": "unknown",
        "dose_value": "unavailable",
        "dose_unit": "unavailable",
        "duration": "unavailable",
        "schedule": _value(row.get("schedule")),
        "route": _value(row.get("route")),
        "exposure_program_ref": "unavailable",
        "occupancy_program_ref": "unavailable",
        "source_provenance": f"OSP model {row['source_model_id']} with repository-generated rollout",
        "missing_fields": "dose_value|dose_unit|duration|execution_program_provenance",
        "missing_reason": "The configured OSP variant stores a library-project dose multiplier, not a provenance-bearing physical execution program.",
        "role": "quarantine",
        "headline_eligible": False,
    } for row in rows]


def _neuron_action_quarantine(rows: Iterable[Mapping[str, str]]) -> list[dict[str, object]]:
    return [{
        "action_uid": row["neuron_rxd_protocol_uid"],
        "candidate_domain": "neuron_rxd",
        "intervention_type": "electrical_current_clamp",
        "agent_set": "not_applicable",
        "target_set": "unknown",
        "dose_value": "0.15",
        "dose_unit": "nA",
        "duration": "120 ms",
        "schedule": "single_current_clamp",
        "route": "not_applicable",
        "exposure_program_ref": "unavailable:not_a_drug_exposure",
        "occupancy_program_ref": "unavailable:not_a_drug_action",
        "source_provenance": row["source_model_id"],
        "missing_fields": "drug_agent_set|drug_target_program|action_contrast_peer",
        "missing_reason": "Physical stimulation protocol is source-backed but has one protocol and no drug-action counterpart for C2 action ablations.",
        "role": "dynamics_only",
        "headline_eligible": False,
    } for row in rows]


def _real_trajectory_candidates(
    repo_root: Path, entries: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    output = []
    for entry in entries:
        manifest = _json(_repo_path(repo_root, str(entry["block_manifest_path"])))
        axes = dict(manifest.get("semantic_axes", {}))
        dose = dict(axes.get("dose", {}))
        time = dict(axes.get("time", {}))
        output.append({
            "candidate_uid": entry["candidate_uid"],
            "source": manifest.get("source_id", "unavailable"),
            "publication_accession": manifest.get("publication_accession_group", "unavailable"),
            "block_hash": manifest.get("block_sha256", "unavailable"),
            "qc_grade": manifest.get("qc_grade", "unavailable"),
            "observation_count": manifest.get("observation_count", 0),
            "effective_sample_size": manifest.get("effective_sample_size", 0),
            "baseline_observation": entry["baseline_status"],
            "structured_action": dose.get("value", "unavailable"),
            "post_action_observation": "available_static_endpoint",
            "time_or_ordered_state": time.get("value", "unavailable"),
            "biological_unit_linkage": dict(manifest.get("biological_unit_policy", {})).get("unit_type", "unavailable"),
            "trajectory_status": entry["trajectory_status"],
            "decision": "not_real_observation_linked_trajectory",
            "reason": "A matched/static endpoint cannot be reinterpreted as a within-unit baseline-to-post-action trajectory.",
        })
    return output


def _independence_graph(
    legacy_rows: list[Mapping[str, str]],
    osp_rows: list[Mapping[str, str]],
    neuron_rows: list[Mapping[str, str]],
    domains: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    counts = Counter(str(row["source_collection"]) for row in legacy_rows)
    nodes = [
        {
            "domain_uid": "legacy_full_microenvironment",
            "simulator_family_uid": domains["legacy_full_microenvironment"]["simulator_family_uid"],
            "trajectory_count": counts["full_microenvironment_teacher"],
            "independence_level": "shared_lineage",
            "provenance": "repository full microenvironment teacher",
        },
        {
            "domain_uid": "legacy_mesoscopic",
            "simulator_family_uid": domains["legacy_mesoscopic"]["simulator_family_uid"],
            "trajectory_count": counts["mesoscopic_solver_teacher"],
            "independence_level": "shared_lineage",
            "provenance": "repository mesoscopic teacher",
        },
        {
            "domain_uid": "osp_pbpk",
            "simulator_family_uid": domains["osp_pbpk"]["simulator_family_uid"],
            "trajectory_count": len(osp_rows),
            "independence_level": "independent",
            "provenance": "public OSP/PK-Sim model library; repository state-grid adapter",
        },
        {
            "domain_uid": "neuron_rxd",
            "simulator_family_uid": domains["neuron_rxd"]["simulator_family_uid"],
            "trajectory_count": len(neuron_rows),
            "independence_level": "independent",
            "provenance": "public ModelDB/NEURON-RxD model; repository microenvironment coupling",
        },
    ]
    graph = {
        "schema_version": "neuronsifter-c2r-dynamics-independence-graph-v1",
        "nodes": nodes,
        "edges": [
            {
                "left": "legacy_full_microenvironment",
                "right": "legacy_mesoscopic",
                "independence_level": "shared_lineage",
                "shared_factors": [
                    "generated_regimen_uid", "generated_parameter_profile_uid", "paired_uid",
                    "repository_action_generation", "repository_training_generator",
                ],
                "supports_simulator_family_holdout": False,
            },
            {
                "left": "osp_pbpk",
                "right": "neuron_rxd",
                "independence_level": "independent",
                "shared_factors": ["repository_state_grid_adapter"],
                "supports_simulator_family_holdout": False,
                "reason": "The raw solvers are independent, but their action semantics and C2 evaluation contracts are not jointly compatible.",
            },
        ],
    }
    decision = {
        "schema_version": "neuronsifter-c2r-independent-domain-decision-v1",
        "status": "not_qualified_for_c2_simulator_holdout",
        "independence_levels_allowed": ["independent", "partially_independent", "shared_lineage", "mirror", "unknown"],
        "raw_independent_domain_count": 2,
        "c2_action_complete_independent_domain_count": 0,
        "required_c2_action_complete_independent_domain_count": 2,
        "simulator_family_holdout_available": False,
        "reason": "OSP and NEURON/RxD are independent model sources, but neither supplies a C2-complete, mutually comparable structured drug-action domain; the legacy full/mesoscopic pair is shared-lineage.",
    }
    return graph, decision


def _baseline_compatibility() -> tuple[dict[str, object], dict[str, object]]:
    rows = [
        ("persistence", "reuse_as_is", "trivial_state_grid", "not_applicable", "blocked_registry_not_sealed"),
        ("linear_shallow_dynamics", "extend_existing", "state_grid_adapter_required", "scalar_or_structured_action_required", "blocked_registry_not_sealed"),
        ("deterministic_neuronsifter", "reuse_as_is", "native_state_grid", "structured_action_program_required", "blocked_registry_not_sealed"),
        ("stochastic_neuronsifter", "reuse_as_is", "native_state_grid", "structured_action_program_required", "blocked_registry_not_sealed"),
        ("drug_id_conditioning", "extend_existing", "shared_state_grid", "drug_id", "blocked_no_action_complete_domain"),
        ("target_id_conditioning", "extend_existing", "shared_state_grid", "target_id", "blocked_no_action_complete_domain"),
        ("scalar_exposure_conditioning", "extend_existing", "shared_state_grid", "physical_scalar_exposure", "blocked_no_observed_exposure_program"),
        ("full_structured_action_conditioning", "extend_existing", "shared_state_grid", "full_structured_action", "blocked_no_action_complete_domain"),
        ("transolver", "extend_existing", "native_transolver_state_grid_adapter", "structured_action_program_required", "blocked_registry_not_sealed"),
        ("diffusionpde", "migrate", "external_bridge_transition_adapter", "structured_action_program_required", "incompatible_no_c2_protocol_adapter"),
        ("wdno", "migrate", "external_bridge_transition_adapter", "structured_action_program_required", "incompatible_no_c2_protocol_adapter"),
    ]
    compatibility = {
        "schema_version": "neuronsifter-c2r-baseline-compatibility-v1",
        "status": "blocked",
        "baselines": [
            {
                "baseline_id": baseline_id,
                "implementation_status": implementation_status,
                "input_adapter": input_adapter,
                "available_action_semantics": action_semantics,
                "output_adapter": "canonical_time_region_compartment_species_grid",
                "shared_split": False,
                "shared_training_budget": False,
                "validation_only_tuning": True,
                "incompatibility_reason": reason,
                "executable_for_c2r_registry": False,
            }
            for baseline_id, implementation_status, input_adapter, action_semantics, reason in rows
        ],
        "strong_baseline_executable": False,
        "reason": "No sealed action-complete registry exists from which equal-split baseline adapters can be instantiated.",
    }
    fairness = {
        "schema_version": "neuronsifter-c2r-baseline-fairness-registry-v1",
        "status": "blocked",
        "fair_action_ablation_available": False,
        "required_shared_contract": [
            "same_trajectory_registry", "same_frozen_manifests", "same_state_grid", "same_action_missingness_masks",
            "same_backbone_capacity_for_action_ablations", "same_optimizer", "same_training_steps",
            "validation_only_checkpoint_selection", "same_tuning_budget", "same_evaluation_budget",
        ],
        "blockers": [
            "no_physical_action_complete_trajectory_domain",
            "no_frozen_action_or_trajectory_holdout_manifest",
            "no_c2_compatible_external_baseline_adapter",
        ],
    }
    return compatibility, fairness


def _options(
    legacy_count: int, osp_count: int, neuron_count: int, real_candidates: list[Mapping[str, object]],
) -> list[dict[str, object]]:
    real_count = sum(int(row["observation_count"]) for row in real_candidates)
    return [
        {
            "option_id": "minimal", "option_name": "minimal", "trajectory_families": "legacy full + mesoscopic",
            "simulator_families": "two shared-lineage repository families", "independence_status": "shared_lineage",
            "trajectory_count": legacy_count, "action_completeness": "0 headline-eligible actions",
            "real_observation_linked_count": 0, "holdouts_enabled": "none",
            "baseline_compatibility": "not fair", "storage": "no additional storage", "compute": "no training allowed",
            "implementation_cost": "low", "scientific_claim_boundary": "none; qualification fails",
            "strict_action_holdout": False, "strict_trajectory_or_simulator_holdout": False,
            "fair_action_ablation": False, "deterministic_stochastic_shared_domain": False,
            "bootstrap_pairing_unit": "not legal for headline", "status": "rejected", "selected": False,
        },
        {
            "option_id": "independent_simulator", "option_name": "independent-simulator", "trajectory_families": "legacy + OSP PBPK + NEURON/RxD",
            "simulator_families": "legacy shared-lineage plus two raw independent sources", "independence_status": "two raw independent, zero C2-compatible",
            "trajectory_count": legacy_count + osp_count + neuron_count, "action_completeness": "OSP incomplete; NEURON single non-drug protocol",
            "real_observation_linked_count": 0, "holdouts_enabled": "none",
            "baseline_compatibility": "not fair across action semantics", "storage": "existing artifacts only", "compute": "adapter and data materialization required",
            "implementation_cost": "high", "scientific_claim_boundary": "no C2 headline claim",
            "strict_action_holdout": False, "strict_trajectory_or_simulator_holdout": False,
            "fair_action_ablation": False, "deterministic_stochastic_shared_domain": False,
            "bootstrap_pairing_unit": "not shared across domains", "status": "rejected", "selected": False,
        },
        {
            "option_id": "observation_anchored", "option_name": "observation-anchored", "trajectory_families": "static public perturbation endpoints + simulator candidates",
            "simulator_families": "not a shared trajectory domain", "independence_status": "not_applicable",
            "trajectory_count": 0, "action_completeness": "physical endpoint actions exist but lack within-unit trajectory linkage",
            "real_observation_linked_count": 0, "holdouts_enabled": "none",
            "baseline_compatibility": "incompatible output axes", "storage": "existing V1/V2 blocks only", "compute": "new domain bridge would be required",
            "implementation_cost": "high", "scientific_claim_boundary": "static perturbation evidence only",
            "strict_action_holdout": False, "strict_trajectory_or_simulator_holdout": False,
            "fair_action_ablation": False, "deterministic_stochastic_shared_domain": False,
            "bootstrap_pairing_unit": "unavailable", "status": "rejected", "selected": False,
        },
    ]


def _gap_decomposition(
    action_complete_count: int, independent: Mapping[str, object], baseline_fair: bool,
) -> dict[str, object]:
    gaps = [
        ("physical_action_completeness", ">= 1 C2 headline domain", action_complete_count, "physical action program with observed exposure/occupancy provenance"),
        ("strict_action_holdout", "available", False, "at least two action programs with complete provenance and frozen grouping"),
        ("strict_trajectory_or_simulator_holdout", "available", False, "two C2-comparable independent trajectory/model domains with frozen overlap audit"),
        ("independent_dynamics_domains", 2, independent["c2_action_complete_independent_domain_count"], "two independent domains with compatible action and state-grid contracts"),
        ("fair_action_ablation", "available", baseline_fair, "one action-complete shared domain and equal baseline adapters"),
        ("deterministic_stochastic_comparison", "available", False, "sealed registry and identical train/validation/test contract"),
        ("strong_baseline_executable", "available", False, "C2-compatible external or Transolver adapter with shared input/output contract"),
        ("bootstrap_pairing_unit", "defined", False, "action/trajectory pairing unit valid across frozen holdout"),
        ("real_observation_linked_domain", "optional but audited", 0, "longitudinal public intervention observation with same-unit baseline and post-action linkage"),
    ]
    return {
        "schema_version": "neuronsifter-c2r-gap-decomposition-v1",
        "status": "blocked",
        "requirements": [
            {
                "requirement_id": requirement_id,
                "required_value": required,
                "current_value": current,
                "shortfall": "unmet" if current != required else "none",
                "blocking_artifacts": [
                    "Output/audits/c2_structured_action_audit.json",
                    "Output/audits/c2_holdout_reachability.json",
                    "Output/audits/c2r_dynamics_independence_graph.json",
                ],
                "minimum_data_change": data_change,
                "minimum_code_change": "none until legal data contract exists" if current in {False, 0} else "registry adapter after data qualification",
                "recoverability": "low" if requirement_id != "real_observation_linked_domain" else "medium",
            }
            for requirement_id, required, current, data_change in gaps
        ],
    }


def _entry_gate(
    contract: Mapping[str, object], immutable: list[Mapping[str, object]], independent: Mapping[str, object],
    action_complete_count: int, baseline_compatibility: Mapping[str, object],
) -> dict[str, object]:
    requirements = {
        "immutable_corpus_identity_verified": all(bool(row["identity_match"]) for row in immutable),
        "physical_action_completeness_passed": action_complete_count > 0,
        "strict_action_holdout_available": False,
        "strict_trajectory_or_simulator_holdout_available": False,
        "independent_dynamics_domains": int(independent["c2_action_complete_independent_domain_count"]),
        "fair_action_ablation_available": False,
        "deterministic_stochastic_comparison_available": False,
        "strong_baseline_executable": bool(baseline_compatibility["strong_baseline_executable"]),
        "bootstrap_pairing_unit_defined": False,
        "c2_leakage_audit_passed": False,
    }
    blockers = [
        {"code": "no_c2_action_complete_trajectory_domain", "evidence": "c2r_action_quarantine.tsv"},
        {"code": "strict_action_holdout_unavailable", "evidence": "c2r_gap_decomposition.json"},
        {"code": "strict_trajectory_or_simulator_holdout_unavailable", "evidence": "c2r_independent_domain_decision.json"},
        {"code": "baseline_fairness_unavailable", "evidence": "c2r_baseline_fairness_registry.json"},
        {"code": "frozen_manifests_not_produced", "evidence": "entry gate fails before manifest construction"},
    ]
    return {
        "schema_version": "neuronsifter-c2r-entry-gate-v1",
        "registry_uid": contract["registry_uid"],
        "status": "pivot_away_from_c2_recommended",
        "training_allowed": False,
        "training_started": False,
        "requirements": requirements,
        "blockers": blockers,
        "registry_sealing": {
            "status": "not_sealed",
            "reason": "No candidate satisfies the C2 trajectory-action and fairness contract.",
            "identity_hash": None,
            "frozen_manifests": "not_produced_gate_failed",
        },
        "claim_boundary": "C2 has implementation and simulator-source inventory evidence only; no structured action, stochastic trajectory, or biological dynamics generalization result is available.",
        "next_route": "pivot_away_from_c2_recommended",
    }


def validate_entry_gate(payload: Mapping[str, object]) -> None:
    allowed = {
        "c2_headline_ready", "c2_simulator_only_reduced_ready", "pivot_away_from_c2_recommended", "blocked_external_dependency",
    }
    if payload.get("registry_uid") != "trajectory_action_registry_v2":
        raise ValueError("C2R entry gate must bind trajectory_action_registry_v2")
    if payload.get("status") not in allowed:
        raise ValueError(f"Unsupported C2R status: {payload.get('status')}")
    if payload.get("training_started") is not False:
        raise ValueError("C2R must not start training")
    if payload.get("status") == "c2_headline_ready" and not payload.get("training_allowed"):
        raise ValueError("headline-ready C2R gate must permit training")
    if payload.get("status") != "c2_headline_ready" and payload.get("training_allowed"):
        raise ValueError("non-headline C2R gate must fail closed")


def _git_value(repo_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=repo_root, check=False, capture_output=True, text=True,
    )
    return completed.stdout.strip()


def _execution_record(payload: Mapping[str, object], repo_root: Path) -> dict[str, object]:
    job_id = os.environ.get("SLURM_JOB_ID", "")
    run_uid = os.environ.get("NRNSIFT_C2R_RUN_UID", "") or f"c2r_qualification_{job_id or 'local'}"
    timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "schema_version": "neuronsifter-c2r-qualification-execution-v1",
        "run_uid": run_uid,
        "job_id": job_id or None,
        "job_name": os.environ.get("SLURM_JOB_NAME", ""),
        "git_commit": _git_value(repo_root, "rev-parse", "HEAD"),
        "dirty_status": "dirty" if _git_value(repo_root, "status", "--porcelain") else "clean",
        "config_hash": payload["contract_hash"],
        "data_hash": payload["input_bindings"]["trajectory_qc_catalog_path"]["sha256"],
        "trajectory_registry_hash": payload["input_bindings"]["trajectory_qc_catalog_path"]["sha256"],
        "start_end": {"completed_at": timestamp},
        "builder_exit_status": "success",
        "terminal_state": "not_finalized",
        "exit_code": None,
        "sacct_available": False,
        "hardware": {
            "hostname": os.uname().nodename,
            "node_list": os.environ.get("SLURM_NODELIST"),
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "gpus": os.environ.get("SLURM_GPUS", "0"),
        },
        "software_environment": {"python_version": sys.version.split()[0]},
        "terminal_evidence_source": "pending_scontrol_finalization" if job_id else "local_process",
        "stdout_stderr": "scheduler output paths are supplied by the submission command",
    }


def finalize_execution(repo_root: Path, job_id: str) -> dict[str, object]:
    config = load_config(repo_root / "Libs/configs/config_stack.yaml")
    audit_root = _repo_path(repo_root, str(config["trajectory_action_registry_c2r"]["outputs"]["audit_root"]))
    path = audit_root / "c2r_qualification_execution.json"
    record = _json(path)
    completed = subprocess.run(
        ["scontrol", "show", "job", str(job_id)], check=False, capture_output=True, text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"scontrol could not finalize C2R job {job_id}: {completed.stderr.strip()}")
    fields = {}
    for token in completed.stdout.replace("\n", " ").split():
        if "=" in token:
            key, value = token.split("=", 1)
            fields[key] = value
    record.update({
        "job_id": str(job_id),
        "job_name": fields.get("JobName", record.get("job_name", "")),
        "terminal_state": fields.get("JobState", "unknown"),
        "exit_code": fields.get("ExitCode"),
        "sacct_available": False,
        "hardware": {
            **dict(record.get("hardware", {})),
            "node_list": fields.get("NodeList", dict(record.get("hardware", {})).get("node_list")),
            "cpus": fields.get("NumCPUs"),
        },
        "terminal_evidence_source": "scontrol_and_preserved_scheduler_logs",
        "scheduler_finalized_at": datetime.now(timezone.utc).isoformat(),
    })
    job_name = str(record.get("job_name") or "NrnSift_C2R_qualification")
    record["stdout_stderr"] = {
        "stdout": f"Output/logs/c2r/{job_name}_{job_id}.stdout.log",
        "stderr": f"Output/logs/c2r/{job_name}_{job_id}.stderr.log",
    }
    write_json(path, record)
    return record


def build(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    repo_root = repo_root.resolve()
    config = load_config(repo_root / "Libs/configs/config_stack.yaml")
    contract = dict(config["trajectory_action_registry_c2r"])
    if not bool(contract["qualification_only"]) or bool(contract["training_permitted"]):
        raise ValueError("C2R config must remain qualification-only with training disabled")
    inputs = dict(contract["inputs"])
    c2_payloads = _load_required_json(repo_root, inputs)
    immutable = _immutable_bindings(repo_root, list(contract["immutable_inputs"]))
    if not all(item["identity_match"] for item in immutable):
        raise ValueError("Frozen V1/V2 identity mismatch blocks C2R qualification")

    legacy_rows = _eligible_legacy_rows(_repo_path(repo_root, str(inputs["trajectory_qc_catalog_path"])))
    regimen_rows = read_tsv(_repo_path(repo_root, str(inputs["regimen_catalog_path"])))
    regimen_by_uid = {str(row["regimen_uid"]): row for row in regimen_rows}
    osp_rows = read_tsv(_repo_path(repo_root, str(inputs["osp_pbpk_catalog_path"])))
    neuron_rows = read_tsv(_repo_path(repo_root, str(inputs["neuron_rxd_catalog_path"])))
    domains = dict(contract["candidate_domains"])
    matrix_rows = _legacy_matrix_rows(legacy_rows, regimen_by_uid, domains)
    matrix_rows.extend(_osp_matrix_rows(osp_rows, domains["osp_pbpk"]))
    matrix_rows.extend(_neuron_matrix_rows(neuron_rows, domains["neuron_rxd"]))

    quarantine_rows = _legacy_action_quarantine(regimen_rows)
    quarantine_rows.extend(_osp_action_quarantine(osp_rows))
    quarantine_rows.extend(_neuron_action_quarantine(neuron_rows))
    real_candidates = _real_trajectory_candidates(repo_root, list(contract["real_observation_candidates"]))
    graph, independent = _independence_graph(legacy_rows, osp_rows, neuron_rows, domains)
    baseline_compatibility, fairness = _baseline_compatibility()
    action_reconstruction = {
        "schema_version": "neuronsifter-c2r-action-provenance-reconstruction-v1",
        "status": "blocked",
        "legacy_c2_action_count": len(regimen_rows),
        "osp_action_count": len(osp_rows),
        "neuron_protocol_count": len(neuron_rows),
        "headline_eligible_action_count": 0,
        "dynamics_only_action_count": len(neuron_rows),
        "quarantined_action_count": len(quarantine_rows) - len(neuron_rows),
        "prohibited_reconstruction_methods": [
            "file_name_inference", "default_config_reverse_inference", "bootstrap_pkpd_prior_as_execution_provenance",
        ],
        "unavailable_is_explicit": True,
        "reason": "The existing C2 drug-action trajectories record synthetic multipliers, not observed physical action programs.",
    }
    gaps = _gap_decomposition(0, independent, bool(fairness["fair_action_ablation_available"]))
    options = _options(len(legacy_rows), len(osp_rows), len(neuron_rows), real_candidates)
    entry = _entry_gate(contract, immutable, independent, 0, baseline_compatibility)
    validate_entry_gate(entry)

    payload = {
        "contract_hash": canonical_sha256(contract),
        "immutable_bindings": immutable,
        "input_bindings": {key: _binding(repo_root, str(value)) for key, value in inputs.items()},
        "c2_source_status": {key: value.get("status") for key, value in c2_payloads.items()},
        "domain_matrix": matrix_rows,
        "gap_decomposition": gaps,
        "action_reconstruction": action_reconstruction,
        "action_quarantine": quarantine_rows,
        "dynamics_graph": graph,
        "independent_domain_decision": independent,
        "real_trajectory_candidates": real_candidates,
        "options": options,
        "baseline_compatibility": baseline_compatibility,
        "baseline_fairness": fairness,
        "entry_gate": entry,
    }
    payload["execution"] = _execution_record(payload, repo_root)
    return payload


def write_outputs(repo_root: Path, payload: Mapping[str, object]) -> None:
    config = load_config(repo_root / "Libs/configs/config_stack.yaml")
    audit_root = _repo_path(repo_root, str(config["trajectory_action_registry_c2r"]["outputs"]["audit_root"]))
    write_tsv(audit_root / "c2r_current_domain_matrix.tsv", MATRIX_FIELDS, list(payload["domain_matrix"]))
    write_json(audit_root / "c2r_gap_decomposition.json", payload["gap_decomposition"])
    write_json(audit_root / "c2r_action_provenance_reconstruction.json", payload["action_reconstruction"])
    write_tsv(audit_root / "c2r_action_quarantine.tsv", QUARANTINE_FIELDS, list(payload["action_quarantine"]))
    write_json(audit_root / "c2r_dynamics_independence_graph.json", payload["dynamics_graph"])
    write_json(audit_root / "c2r_independent_domain_decision.json", payload["independent_domain_decision"])
    write_tsv(audit_root / "c2r_real_trajectory_candidate_audit.tsv", REAL_CANDIDATE_FIELDS, list(payload["real_trajectory_candidates"]))
    write_tsv(audit_root / "c2r_registry_option_matrix.tsv", OPTION_FIELDS, list(payload["options"]))
    write_json(audit_root / "c2r_baseline_compatibility.json", payload["baseline_compatibility"])
    write_json(audit_root / "c2r_baseline_fairness_registry.json", payload["baseline_fairness"])
    write_json(audit_root / "c2r_entry_gate_decision.json", payload["entry_gate"])
    write_json(audit_root / "c2r_qualification_execution.json", payload["execution"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Write C2R audit artifacts after the fail-closed qualification.")
    parser.add_argument("--finalize-job", default="", help="Record final scheduler state for an already-written C2R qualification job.")
    args = parser.parse_args()
    if args.finalize_job:
        record = finalize_execution(REPO_ROOT, args.finalize_job)
        print(f"c2r_job_terminal_state={record['terminal_state']}")
        return
    payload = build(REPO_ROOT)
    if args.write:
        write_outputs(REPO_ROOT, payload)
    print(f"c2r_registry_uid=trajectory_action_registry_v2")
    print(f"c2r_entry_status={payload['entry_gate']['status']}")
    print(f"c2r_training_allowed={payload['entry_gate']['training_allowed']}")


if __name__ == "__main__":
    main()
