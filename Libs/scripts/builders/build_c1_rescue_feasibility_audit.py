"""Build the fail-closed C1 rescue feasibility and recovery-option artifacts."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.public_foundation_evidence import safe_repo_path, sha256_file  # noqa: E402
from Libs.utils.data.raw_data_registry import read_tsv, write_json  # noqa: E402


CONFIG_PATH = "Libs/configs/data/public_foundation_corpus_rescue.yaml"
RESCUE_BUNDLE_PATH = "Output/audits/public_foundation_corpus_rescue_evidence_bundle.json"
RESCUE_VALIDATION_PATH = "Output/audits/public_foundation_corpus_rescue_evidence_validation.json"
RESCUE_BLOCKS_PATH = "Output/audits/public_foundation_corpus_rescue_canonical_observation_blocks.json"
RESCUE_SPLITS_PATH = "Output/audits/public_foundation_corpus_rescue_frozen_splits.json"
RESCUE_QC_PATH = "Output/audits/public_foundation_corpus_rescue_qc_ess_statistics.json"
RESCUE_STORAGE_PATH = "Output/audits/public_foundation_corpus_rescue_storage_ledger.json"
RESCUE_OBJECTIVE_CAPABILITY_PATH = "Output/audits/c1_rescue_objective_capability_manifest.json"
RESCUE_UNIFIED_PREFLIGHT_PATH = "Output/audits/c1_rescue_preflight.json"
PRIMARY_BLOCKS_PATH = "Output/audits/public_foundation_corpus_canonical_observation_blocks.json"
PRIMARY_PREFLIGHT_PATH = "Output/audits/c1_preflight.json"
PRIMARY_INVENTORY_PATH = "Output/audits/c1_repository_inventory.json"
PRIMARY_BASELINES_PATH = "Output/audits/c1_baseline_registry.json"
PRIMARY_JOB_MATRIX_PATH = "Output/audits/c1_job_matrix.json"
PRIMARY_G3_PATH = "Output/audits/c1_g3_validation.json"
PRIMARY_GATE_PATH = "Output/paper_reports/c1_gate_summary.json"
ANCHOR_PATH = "Output/audits/c1_rescue_anchor_probe.json"
CAPABILITY_PATH = "Output/audits/c1_rescue_current_capability_matrix.json"
GAP_PATH = "Output/audits/c1_rescue_gap_decomposition.json"
CANDIDATE_PATH = "Output/audits/c1_rescue_candidate_source_audit.tsv"
INDEPENDENCE_PATH = "Output/audits/c1_rescue_candidate_independence_graph.json"
OPTION_PATH = "Output/audits/c1_rescue_option_matrix.tsv"
DECISION_PATH = "Output/audits/c1_rescue_feasibility_decision.json"
ROUTE_CLOSEOUT_PATH = "Output/audits/c1_route_closeout.json"
CLAIM_SCOPE_PATH = "Output/paper_reports/c1_claim_scope.json"
DEX_REGIMEN_UID = "canonical_compound_regimen:pubchem:5743"


CANDIDATE_FIELDS = [
    "candidate_uid",
    "official_source",
    "official_accession",
    "official_release",
    "license_terms_snapshot",
    "training_role_compatibility",
    "raw_value_matrix_availability",
    "metadata_completeness",
    "biological_unit_definition",
    "expected_grade",
    "expected_ess",
    "task_family_enabled",
    "holdout_family_enabled",
    "cross_source_anchor_potential",
    "action_linkage_potential",
    "storage_estimate_bytes",
    "parser_availability",
    "implementation_cost",
    "scientific_risk",
    "publication_accession_group",
    "mirror_group",
    "underlying_matrix_identity",
    "processing_lineage",
    "content_hash",
    "decision",
    "provenance",
]

OPTION_FIELDS = [
    "option_id",
    "option_name",
    "status",
    "new_sources_accessions",
    "new_canonical_blocks",
    "new_expected_observations",
    "new_expected_ess",
    "new_task_families",
    "new_holdout_families",
    "new_actions",
    "new_cross_source_anchors",
    "new_parsers",
    "new_storage_requirement_bytes",
    "new_hpc_requirement",
    "license_risk",
    "estimated_scientific_sufficiency",
    "full_objective_reachable_task_families",
    "full_objective_reachable_holdout_families",
    "meets_task_family_contract",
    "meets_holdout_family_contract",
    "meets_cross_source_anchor_contract",
    "meets_perturbation_anchor_contract",
    "selected",
    "provenance",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: str, repo_root: Path = REPO_ROOT) -> dict[str, object]:
    payload = json.loads(safe_repo_path(repo_root, path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _policy(repo_root: Path = REPO_ROOT) -> Mapping[str, object]:
    payload = yaml.safe_load(safe_repo_path(repo_root, CONFIG_PATH).read_text(encoding="utf-8"))
    policy = payload.get("public_foundation_corpus_rescue")
    if not isinstance(policy, Mapping):
        raise ValueError("C1 rescue policy is malformed")
    rescue = policy.get("rescue_feasibility")
    if not isinstance(rescue, Mapping):
        raise ValueError("C1 rescue feasibility policy is absent")
    return policy


def _write_tsv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _json_cell(value: object) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _artifact_ref(path: str, repo_root: Path = REPO_ROOT) -> dict[str, object]:
    target = safe_repo_path(repo_root, path)
    return {
        "path": path,
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
    }


def _split_roles(
    split_artifact: Mapping[str, object], split_uid: str, repo_root: Path
) -> dict[str, list[dict[str, str]]]:
    manifest = next(
        row for row in split_artifact["manifests"] if row["split_uid"] == split_uid
    )
    roles: dict[str, list[dict[str, str]]] = {}
    for role, binding in manifest["role_bindings"].items():
        path = safe_repo_path(repo_root, str(binding["path"]))
        if sha256_file(path) != binding["sha256"]:
            raise ValueError(f"Frozen split role drift: {split_uid}:{role}")
        rows = read_tsv(path)
        if len(rows) != int(binding["record_count"]):
            raise ValueError(f"Frozen split role count drift: {split_uid}:{role}")
        roles[str(role)] = rows
    return roles


def _family_count(rows: Sequence[Mapping[str, str]], family_id: str) -> int:
    return sum(row.get("family_uid") == family_id for row in rows)


def _manifest_bytes(manifest: Mapping[str, object]) -> int:
    records = manifest.get("block_files", [])
    return sum(int(row.get("bytes", 0)) for row in records if isinstance(row, Mapping))


def _full_objective_reachability(
    capability: Mapping[str, object],
) -> tuple[list[str], list[str], list[dict[str, object]]]:
    mappings = {
        "family_holdout_cns_cell_type_region_atlas_transcriptomics": (
            "atlas_representation_transfer",
            "family_holdout",
        ),
        "family_holdout_cns_disease_aging_state_transcriptomics": (
            "disease_state_transfer",
            "family_holdout",
        ),
        "family_holdout_compound_dose_time_perturbation_transcriptomics": (
            "perturbation_action_transfer",
            "family_holdout",
        ),
        "source_publication_holdout": (
            "perturbation_action_transfer",
            "source_publication_holdout",
        ),
        "regimen_holdout": (
            "perturbation_action_transfer",
            "regimen_holdout",
        ),
        "target_program_holdout": (
            "perturbation_action_transfer",
            "target_program_holdout",
        ),
    }
    splits = capability.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("C1 rescue objective capability manifest has no split scopes")
    tasks: set[str] = set()
    holdouts: set[str] = set()
    rows: list[dict[str, object]] = []
    for split_uid, (task_family, holdout_family) in mappings.items():
        split = splits.get(split_uid)
        if not isinstance(split, Mapping):
            raise ValueError(f"C1 rescue objective capability split is missing: {split_uid}")
        roles = split.get("roles")
        if not isinstance(roles, Mapping):
            raise ValueError(f"C1 rescue objective capability roles are missing: {split_uid}")
        role_status: dict[str, str] = {}
        role_blockers: dict[str, object] = {}
        for role in ("train", "validation"):
            role_scope = roles.get(role)
            all_sources = (
                role_scope.get("all_sources")
                if isinstance(role_scope, Mapping)
                else None
            )
            profiles = (
                all_sources.get("profile_sampler_structural_feasibility")
                if isinstance(all_sources, Mapping)
                else None
            )
            profile = (
                profiles.get("neuronsifter_foundation_state_encoder")
                if isinstance(profiles, Mapping)
                else None
            )
            if not isinstance(profile, Mapping):
                raise ValueError(
                    f"C1 rescue full-objective scope is missing: {split_uid}:{role}"
                )
            role_status[role] = str(profile.get("status"))
            role_blockers[role] = profile.get("blocker")
        reachable = all(
            role_status[role] == "structurally_feasible"
            for role in ("train", "validation")
        )
        if reachable:
            tasks.add(task_family)
            holdouts.add(holdout_family)
        rows.append(
            {
                "split_uid": split_uid,
                "task_family": task_family,
                "holdout_family": holdout_family,
                "train_status": role_status["train"],
                "validation_status": role_status["validation"],
                "train_blocker": role_blockers["train"],
                "validation_blocker": role_blockers["validation"],
                "full_objective_train_validation_reachable": reachable,
            }
        )
    return sorted(tasks), sorted(holdouts), rows


def _candidate_rows(
    policy: Mapping[str, object],
    blocks: Mapping[str, object],
    primary_blocks: Mapping[str, object],
    repo_root: Path,
) -> list[dict[str, object]]:
    rescue = policy["rescue_feasibility"]
    candidate_policy = rescue["candidate_audit"]
    all_rows = [*blocks["qualified_blocks"], *primary_blocks.get("blocks", [])]
    by_uid = {str(row["canonical_uid"]): row for row in all_rows}
    result: list[dict[str, object]] = []
    for candidate in candidate_policy.values():
        block = by_uid[str(candidate["block_uid"])]
        manifest_path = safe_repo_path(repo_root, str(block.get("manifest_path", "")))
        manifest = _load_json(manifest_path.relative_to(repo_root).as_posix(), repo_root) if manifest_path.is_file() else {}
        release = manifest.get("source_release", {}) if isinstance(manifest, Mapping) else {}
        terms = release.get("terms_snapshot", {}) if isinstance(release, Mapping) else {}
        if candidate["evidence_kind"] == "migration_input_only":
            license_terms = str(candidate["license_terms_status"])
            storage = int(block.get("value_bytes", 0))
            provenance = PRIMARY_BLOCKS_PATH
        else:
            license_terms = _json_cell(terms or candidate["license_terms_status"])
            storage = _manifest_bytes(manifest)
            provenance = str(block["manifest_path"])
        result.append(
            {
                "candidate_uid": candidate["candidate_uid"],
                "official_source": candidate["official_source"],
                "official_accession": candidate["official_accession"],
                "official_release": candidate["official_release"],
                "license_terms_snapshot": license_terms,
                "training_role_compatibility": candidate["training_role_compatibility"],
                "raw_value_matrix_availability": bool(
                    manifest.get("real_value_matrix") is True
                    or block.get("value_block_path")
                ),
                "metadata_completeness": candidate["metadata_completeness"],
                "biological_unit_definition": candidate["biological_unit_definition"],
                "expected_grade": candidate["expected_grade"],
                "expected_ess": candidate["expected_ess"],
                "task_family_enabled": candidate["task_family_enabled"],
                "holdout_family_enabled": candidate["holdout_family_enabled"],
                "cross_source_anchor_potential": candidate["cross_source_anchor_potential"],
                "action_linkage_potential": candidate["action_linkage_potential"],
                "storage_estimate_bytes": storage,
                "parser_availability": candidate["parser_availability"],
                "implementation_cost": candidate["implementation_cost"],
                "scientific_risk": candidate["scientific_risk"],
                "publication_accession_group": block.get("publication_accession_group", ""),
                "mirror_group": block.get("mirror_group", ""),
                "underlying_matrix_identity": block.get("block_sha256", "migration_input_only"),
                "processing_lineage": candidate["evidence_kind"],
                "content_hash": block.get("block_sha256", ""),
                "decision": candidate["decision"],
                "provenance": provenance,
            }
        )
    return result


def _gap(
    requirement_id: str,
    required_value: object,
    current_value: object,
    shortfall: object,
    blocking_artifacts: Sequence[str],
    minimum_data_change: str,
    minimum_code_change: str,
    recoverability: str,
) -> dict[str, object]:
    if recoverability not in {"high", "medium", "low", "non_recoverable"}:
        raise ValueError(f"Unsupported recoverability: {recoverability}")
    return {
        "requirement_id": requirement_id,
        "required_value": required_value,
        "current_value": current_value,
        "shortfall": shortfall,
        "blocking_artifacts": list(blocking_artifacts),
        "minimum_data_change": minimum_data_change,
        "minimum_code_change": minimum_code_change,
        "recoverability": recoverability,
    }


def build(repo_root: Path = REPO_ROOT) -> dict[str, object]:
    repo_root = repo_root.resolve()
    policy = _policy(repo_root)
    rescue = policy["rescue_feasibility"]
    bundle = _load_json(RESCUE_BUNDLE_PATH, repo_root)
    validation = _load_json(RESCUE_VALIDATION_PATH, repo_root)
    blocks = _load_json(RESCUE_BLOCKS_PATH, repo_root)
    splits = _load_json(RESCUE_SPLITS_PATH, repo_root)
    qc = _load_json(RESCUE_QC_PATH, repo_root)
    storage = _load_json(RESCUE_STORAGE_PATH, repo_root)
    objective_capability = _load_json(RESCUE_OBJECTIVE_CAPABILITY_PATH, repo_root)
    primary_blocks = _load_json(PRIMARY_BLOCKS_PATH, repo_root)
    primary_preflight = _load_json(PRIMARY_PREFLIGHT_PATH, repo_root)
    inventory = _load_json(PRIMARY_INVENTORY_PATH, repo_root)
    baselines = _load_json(PRIMARY_BASELINES_PATH, repo_root)
    job_matrix = _load_json(PRIMARY_JOB_MATRIX_PATH, repo_root)
    g3 = _load_json(PRIMARY_G3_PATH, repo_root)
    gate = _load_json(PRIMARY_GATE_PATH, repo_root)
    anchor = _load_json(ANCHOR_PATH, repo_root)

    parent = policy["parent_corpus"]
    for key in ("evidence_bundle", "canonical_blocks", "frozen_splits", "leakage_audit"):
        path = safe_repo_path(repo_root, str(parent[f"{key}_path"]))
        if sha256_file(path) != parent[f"{key}_sha256"]:
            raise ValueError(f"Frozen primary C1 artifact drift: {key}")
    if (
        bundle.get("status") != "valid_fail_closed"
        or validation.get("status") != "valid_fail_closed"
        or validation.get("blocker_count") != 0
        or bundle.get("corpus_hash") != validation.get("corpus_identity_hash")
    ):
        raise ValueError("C1 rescue evidence is not in the sealed valid_fail_closed state")

    source_roles = _split_roles(splits, "source_publication_holdout", repo_root)
    atlas_roles = _split_roles(
        splits, "family_holdout_cns_cell_type_region_atlas_transcriptomics", repo_root
    )
    regimen_roles = _split_roles(splits, "regimen_holdout", repo_root)
    dex_by_role = {
        role: [row for row in rows if row["regimen_uid"] == DEX_REGIMEN_UID]
        for role, rows in source_roles.items()
    }
    source_anchor_reachable = (
        {row["source_id"] for row in dex_by_role["train"]} == {"clue_lincs"}
        and {row["source_id"] for row in dex_by_role["test"]} == {"geo"}
        and len(dex_by_role["train"]) > 0
        and len(dex_by_role["test"]) == 1
    )
    atlas_test_units = _family_count(
        atlas_roles["test"], "cns_cell_type_region_atlas_transcriptomics"
    )
    atlas_sources = sorted({row["source_id"] for row in atlas_roles["test"]})
    atlas_reachable = atlas_test_units == 9 and len(atlas_sources) == 2
    regimen_reachable = (
        _family_count(regimen_roles["train"], "compound_dose_time_perturbation_transcriptomics")
        > 0
        and _family_count(
            regimen_roles["test"], "compound_dose_time_perturbation_transcriptomics"
        )
        > 0
    )
    reachable_tasks = []
    if atlas_reachable:
        reachable_tasks.append("atlas_representation_transfer")
    if source_anchor_reachable:
        reachable_tasks.append("perturbation_action_transfer")
    reachable_holdouts = []
    if atlas_reachable:
        reachable_holdouts.append("family_holdout")
    if source_anchor_reachable:
        reachable_holdouts.append("source_publication_holdout")
    (
        full_objective_tasks,
        full_objective_holdouts,
        full_objective_scope_audit,
    ) = _full_objective_reachability(objective_capability)

    block_rows = blocks["qualified_blocks"]
    atlas_blocks = [
        row["canonical_uid"]
        for row in block_rows
        if row["family_id"] == "cns_cell_type_region_atlas_transcriptomics"
    ]
    perturbation_blocks = [
        row["canonical_uid"]
        for row in block_rows
        if row["family_id"] == "compound_dose_time_perturbation_transcriptomics"
    ]
    primary_missing = {
        str(row.get("capability")): str(row.get("reason", ""))
        for row in primary_preflight.get("headline_requirements", [])
        if row.get("code") == "headline_capability_incomplete"
    }
    implementation = {
        capability: {
            "status": (
                "complete"
                if capability == "paired_bootstrap_statistics"
                else "incomplete"
            ),
            "reason": (
                "paired unit bootstrap and deterministic permutation utilities are tested"
                if capability == "paired_bootstrap_statistics"
                else primary_missing.get(capability, "C1-specific executable is not registered")
            ),
        }
        for capability in rescue["required_headline_capabilities"]
    }
    missing_capabilities = [
        name for name, row in implementation.items() if row["status"] != "complete"
    ]

    capability = {
        "schema_version": "neuronsifter-c1-rescue-current-capability-v1",
        "status": "passed_data_reachability_implementation_incomplete",
        "created_at": _now(),
        "v1_corpus_uid": parent["corpus_uid"],
        "v1_corpus_identity_hash": parent["identity_hash"],
        "v2_corpus_uid": bundle["corpus_uid"],
        "v2_corpus_identity_hash": bundle["corpus_hash"],
        "available_task_families": full_objective_tasks,
        "available_holdout_families": full_objective_holdouts,
        "data_reachable_task_families": reachable_tasks,
        "data_reachable_holdout_families": reachable_holdouts,
        "maximum_full_objective_task_family_count": len(full_objective_tasks),
        "maximum_full_objective_holdout_family_count": len(
            full_objective_holdouts
        ),
        "full_objective_scope_audit": full_objective_scope_audit,
        "additional_structurally_available_holdouts": (
            ["regimen_holdout"] if regimen_reachable else []
        ),
        "qualified_blocks_supporting_each_task": {
            "atlas_representation_transfer": atlas_blocks,
            "perturbation_action_transfer": perturbation_blocks,
            "disease_state_transfer": [
                row["canonical_uid"]
                for row in block_rows
                if row["family_id"] == "cns_disease_aging_state_transcriptomics"
            ],
        },
        "qualified_biological_units": {
            family: int(stats["grade_ab_effective_sample_size"])
            for family, stats in qc["family_statistics"].items()
        },
        "eligible_labels": {
            "atlas_representation_transfer": [
                "donor_uid",
                "source_accession",
                "cell_context_and_region_metadata_when_observed",
            ],
            "perturbation_action_transfer": [
                "qualified_action_uid",
                "canonical_compound_uid",
                "dose",
                "duration",
                "source_accession",
            ],
            "disease_state_transfer": ["Alzheimer_disease", "normal_control"],
        },
        "action_linked_observations": int(bundle["qualified_action_count"]),
        "cross_source_anchors": [
            {
                "anchor_id": "pubchem:5743",
                "canonical_agent_name": "dexamethasone",
                "qualified_action_count": bundle["perturbation_transfer_anchor"][
                    "qualified_action_count"
                ],
                "train_sources": sorted(
                    {row["source_id"] for row in dex_by_role["train"]}
                ),
                "validation_sources": sorted(
                    {row["source_id"] for row in dex_by_role["validation"]}
                ),
                "test_sources": sorted({row["source_id"] for row in dex_by_role["test"]}),
                "split_uid": "source_publication_holdout",
                "status": "reachable" if source_anchor_reachable else "blocked",
            }
        ],
        "missing_evaluation_heads": missing_capabilities,
        "missing_diagnostic_capabilities": [
            "independent_source_probe",
            "independent_publication_accession_probe",
            "independent_family_probe",
            "independent_block_probe",
            "split_role_leakage_probe",
            "source_family_block_label_permutations",
        ],
        "hard_data_blockers": [
            {
                "task_family": "disease_state_transfer",
                "code": "no_region_matched_cross_source_disease_anchor",
                "impact": "disease labels remain confounded with cohort and brain region",
            }
        ],
        "implementation_only_blockers": missing_capabilities,
        "headline_evaluation_capability": "incomplete",
        "headline_training_allowed": False,
        "empirical_transfer_evidence": "unavailable",
        "g3_status": gate["status"],
        "source_evidence": {
            "inventory": _artifact_ref(PRIMARY_INVENTORY_PATH, repo_root),
            "baseline_registry": _artifact_ref(PRIMARY_BASELINES_PATH, repo_root),
            "v1_job_matrix": _artifact_ref(PRIMARY_JOB_MATRIX_PATH, repo_root),
            "v1_g3": _artifact_ref(PRIMARY_G3_PATH, repo_root),
            "v2_evidence": _artifact_ref(RESCUE_BUNDLE_PATH, repo_root),
            "v2_validation": _artifact_ref(RESCUE_VALIDATION_PATH, repo_root),
        },
    }
    write_json(safe_repo_path(repo_root, CAPABILITY_PATH), capability)

    gaps = [
        _gap(
            "task_family_count",
            ">=2",
            len(full_objective_tasks),
            max(0, 2 - len(full_objective_tasks)),
            [CAPABILITY_PATH, RESCUE_OBJECTIVE_CAPABILITY_PATH],
            (
                "add an independent public block that makes a second frozen "
                "split train/validation scope satisfy the full objective contract"
            ),
            (
                "a new preregistered task-specific objective and fairness contract "
                "would be required; silent objective removal is prohibited"
            ),
            "low",
        ),
        _gap(
            "holdout_family_count",
            ">=2",
            len(full_objective_holdouts),
            max(0, 2 - len(full_objective_holdouts)),
            [RESCUE_SPLITS_PATH, CAPABILITY_PATH, RESCUE_OBJECTIVE_CAPABILITY_PATH],
            (
                "add public independent support for a second frozen holdout whose "
                "train/validation roles meet the full objective contract"
            ),
            (
                "a new preregistered holdout-specific objective and fairness "
                "contract would be required"
            ),
            "low",
        ),
        _gap(
            "qualified_cross_source_anchor_count",
            ">=1",
            int(bundle["qualified_cross_source_anchor_count"]),
            0,
            [RESCUE_BUNDLE_PATH, RESCUE_SPLITS_PATH],
            "none",
            "none for data qualification",
            "high",
        ),
        _gap(
            "qualified_perturbation_transfer_anchor_count",
            ">=1",
            int(bundle["qualified_perturbation_transfer_anchor_count"]),
            0,
            [RESCUE_BUNDLE_PATH, ANCHOR_PATH],
            "none",
            "none for data qualification",
            "high",
        ),
        _gap(
            "headline_evaluation_capability",
            "complete",
            "incomplete",
            missing_capabilities,
            [PRIMARY_PREFLIGHT_PATH, PRIMARY_BASELINES_PATH],
            "none",
            "implement validation selection, task evaluators, baseline runner, diagnostics, and result ingestion",
            "medium",
        ),
        _gap(
            "source_shortcut_diagnostic_capability",
            "complete",
            "incomplete",
            capability["missing_diagnostic_capabilities"],
            [PRIMARY_INVENTORY_PATH],
            "none",
            "implement independent probes and three registered label permutations",
            "high",
        ),
        _gap(
            "few_shot_capability",
            "complete",
            "incomplete",
            ["linear_probe", "1_shot", "5_shot", "10_shot", "full_fine_tune"],
            [PRIMARY_BASELINES_PATH],
            "none",
            "implement unit-level feasibility and shared draw manifests",
            "high",
        ),
        _gap(
            "scaling_capability",
            "complete",
            "incomplete",
            ["unit_fraction_runner", "family_diversity_runner", "objective_scale_runner"],
            [PRIMARY_JOB_MATRIX_PATH],
            "none",
            "implement deterministic unit-level scaling plans and result aggregation",
            "medium",
        ),
        _gap(
            "paired_bootstrap_domain",
            "available",
            "available",
            0,
            ["Libs/utils/evaluation/foundation_state_c1_metrics.py"],
            "none",
            "none; paired biological-unit bootstrap is implemented",
            "high",
        ),
    ]
    gap_payload = {
        "schema_version": "neuronsifter-c1-rescue-gap-decomposition-v1",
        "status": "passed",
        "created_at": _now(),
        "v1_corpus_identity_hash": parent["identity_hash"],
        "v2_corpus_identity_hash": bundle["corpus_hash"],
        "requirements": gaps,
        "data_gap_count": sum(bool(row["shortfall"]) for row in gaps[:4]),
        "implementation_gap_count": sum(bool(row["shortfall"]) for row in gaps[4:]),
        "blockers": [],
    }
    write_json(safe_repo_path(repo_root, GAP_PATH), gap_payload)

    candidates = _candidate_rows(policy, blocks, primary_blocks, repo_root)
    _write_tsv(safe_repo_path(repo_root, CANDIDATE_PATH), CANDIDATE_FIELDS, candidates)
    independence = {
        "schema_version": "neuronsifter-c1-rescue-candidate-independence-graph-v1",
        "status": "passed",
        "created_at": _now(),
        "node_count": len(candidates) + 2,
        "nodes": [
            {
                "candidate_uid": row["candidate_uid"],
                "source_accession": row["official_accession"],
                "publication_accession_group": row["publication_accession_group"],
                "mirror_group": row["mirror_group"],
                "content_hash": row["content_hash"],
                "decision": row["decision"],
            }
            for row in candidates
        ]
        + [
            {
                "candidate_uid": "v1_clue_dexamethasone_anchor",
                "source_accession": "GSE92742-TouchstoneV1.1::NPC_NEU_TRT_CP",
                "publication_accession_group": "clue::GSE92742-TouchstoneV1.1::NPC_NEU_TRT_CP",
                "mirror_group": "clue::GSE92742-TouchstoneV1.1::NPC_NEU_TRT_CP",
                "content_hash": anchor["qualified_clue_dexamethasone_anchor"]["action_sha256"],
                "decision": "frozen_v1_anchor",
            },
            {
                "candidate_uid": "v1_e_mtab_6772_treated_block",
                "source_accession": "E-MTAB-6772",
                "publication_accession_group": "doi:10.1007/s13311-019-00754-z::E-MTAB-6772",
                "mirror_group": "arrayexpress::E-MTAB-6772",
                "content_hash": policy["candidates"]["e_mtab_6772_controls"][
                    "parent_v1_block_sha256"
                ],
                "decision": "frozen_v1_block",
            },
        ],
        "edges": [
            {
                "left": "perturbation_geo_gse262992_dexamethasone",
                "right": "v1_clue_dexamethasone_anchor",
                "relation": "independent_cross_source_compound_anchor",
                "publication_accession_independent": True,
                "mirror_independent": True,
                "underlying_matrix_independent": True,
                "biological_unit_independent": True,
                "study_cohort_independent": True,
                "processing_lineage_independent": True,
                "content_hash_independent": True,
                "counts_as_cross_source_anchor": True,
            },
            {
                "left": "perturbation_arrayexpress_e_mtab_6772_controls_v2",
                "right": "v1_e_mtab_6772_treated_block",
                "relation": "same_accession_and_underlying_study_matrix",
                "publication_accession_independent": False,
                "mirror_independent": False,
                "underlying_matrix_independent": False,
                "biological_unit_independent": True,
                "study_cohort_independent": False,
                "processing_lineage_independent": False,
                "content_hash_independent": True,
                "counts_as_cross_source_anchor": False,
            },
        ],
        "same_matrix_counted_as_independent": False,
        "blockers": [],
    }
    write_json(safe_repo_path(repo_root, INDEPENDENCE_PATH), independence)

    options: list[dict[str, object]] = []
    for option_id, option in rescue["options"].items():
        task_count = len(full_objective_tasks)
        holdout_count = len(full_objective_holdouts)
        option_meets_execution = (
            task_count >= int(rescue["required_task_family_count"])
            and holdout_count >= int(rescue["required_holdout_family_count"])
        )
        options.append(
            {
                "option_id": option_id,
                "option_name": option["option_name"],
                "status": (
                    option["status"]
                    if option_meets_execution
                    else "materialized_but_rejected_for_c1_rescue"
                ),
                "new_sources_accessions": _json_cell(option["new_sources_accessions"]),
                "new_canonical_blocks": _json_cell(option["new_canonical_blocks"]),
                "new_expected_observations": option["new_expected_observations"],
                "new_expected_ess": option["new_expected_ess"],
                "new_task_families": _json_cell(option["new_task_families"]),
                "new_holdout_families": _json_cell(option["reachable_holdout_families"]),
                "new_actions": option["new_actions"],
                "new_cross_source_anchors": option["new_cross_source_anchors"],
                "new_parsers": _json_cell(option["new_parsers"]),
                "new_storage_requirement_bytes": storage["v2_increment_bytes"],
                "new_hpc_requirement": "completed_cpu03_materialization_and_evidence_sealing",
                "license_risk": option["license_risk"],
                "estimated_scientific_sufficiency": (
                    option["estimated_scientific_sufficiency"]
                    if option_meets_execution
                    else "insufficient_full_objective_task_and_holdout_reachability"
                ),
                "full_objective_reachable_task_families": _json_cell(
                    full_objective_tasks
                ),
                "full_objective_reachable_holdout_families": _json_cell(
                    full_objective_holdouts
                ),
                "meets_task_family_contract": task_count
                >= int(rescue["required_task_family_count"]),
                "meets_holdout_family_contract": holdout_count
                >= int(rescue["required_holdout_family_count"]),
                "meets_cross_source_anchor_contract": option["new_cross_source_anchors"]
                >= int(rescue["required_cross_source_anchor_count"]),
                "meets_perturbation_anchor_contract": bundle[
                    "qualified_perturbation_transfer_anchor_count"
                ]
                >= int(rescue["required_perturbation_transfer_anchor_count"]),
                "selected": (
                    option_id == rescue["selected_option_id"]
                    and option_meets_execution
                ),
                "provenance": RESCUE_BUNDLE_PATH,
            }
        )
    _write_tsv(safe_repo_path(repo_root, OPTION_PATH), OPTION_FIELDS, options)

    conditions = {
        "public_and_role_compatible": all(
            row["decision"] != "admitted_rescue"
            or row["training_role_compatibility"] == "qualified_for_mainline"
            for row in candidates
        ),
        "real_value_matrix_available": any(
            row["decision"] == "admitted_rescue"
            and row["raw_value_matrix_availability"] is True
            for row in candidates
        ),
        "independence_verifiable": source_anchor_reachable
        and independence["same_matrix_counted_as_independent"] is False,
        "grade_ab_plausible": all(
            row["expected_grade"] in {"A", "B"}
            for row in candidates
            if str(row["decision"]).startswith("admitted")
        ),
        "ess_positive": int(bundle["grade_ab_effective_sample_size"]) > 0,
        "two_task_families_reachable": len(full_objective_tasks)
        >= int(rescue["required_task_family_count"]),
        "two_holdout_families_reachable": len(full_objective_holdouts)
        >= int(rescue["required_holdout_family_count"]),
        "cross_source_anchor_reachable": source_anchor_reachable,
        "perturbation_transfer_anchor_reachable": bundle[
            "qualified_perturbation_transfer_anchor_count"
        ]
        >= int(rescue["required_perturbation_transfer_anchor_count"]),
        "storage_cap_remains_valid": storage["cap_status"] == "passed"
        and storage["partial_scan"] is False,
    }
    decision_status = (
        "rescue_feasible" if all(conditions.values()) else "pivot_to_c2_recommended"
    )
    unified_path = safe_repo_path(repo_root, RESCUE_UNIFIED_PREFLIGHT_PATH)
    unified = _load_json(RESCUE_UNIFIED_PREFLIGHT_PATH, repo_root) if unified_path.is_file() else {}
    rescue_run_results = []
    for result_path in sorted(
        (repo_root / "Output/results/c1_rescue/runs").glob("*/run_result.json")
    ):
        result = _load_json(result_path.relative_to(repo_root).as_posix(), repo_root)
        rescue_run_results.append(
            {
                "path": result_path.relative_to(repo_root).as_posix(),
                "run_uid": result.get("run_uid"),
                "model_scale": result.get("model_scale"),
                "status": result.get("status"),
                "training_started": result.get("training_started"),
                "headline": result.get("headline"),
                "checkpoint_hash": result.get("checkpoint_hash"),
            }
        )
    closeout_artifacts: dict[str, dict[str, object]] = {}
    if decision_status == "pivot_to_c2_recommended":
        route_closeout = {
            "schema_version": "neuronsifter-c1-route-closeout-v1",
            "status": "pivot_to_c2_recommended",
            "created_at": _now(),
            "c1_implementation_capability": "implemented_and_smoke_validated",
            "c1_smoke_capability": (
                "v1_and_v2_development_smoke_only"
                if any(row["status"] == "passed" for row in rescue_run_results)
                else "v1_development_smoke_only"
            ),
            "c1_empirical_transfer_evidence": "unavailable",
            "g3_status": "blocked_incomplete",
            "v1_corpus_status": "frozen_unchanged",
            "v1_corpus_identity_hash": parent["identity_hash"],
            "v2_corpus_status": "sealed_valid_fail_closed_not_c1_ready",
            "v2_corpus_identity_hash": bundle["corpus_hash"],
            "maximum_full_objective_task_family_count": len(
                full_objective_tasks
            ),
            "maximum_full_objective_holdout_family_count": len(
                full_objective_holdouts
            ),
            "foundation_claim_permission": "prohibited",
            "student_smoke_checkpoint_release_allowed": False,
            "c2_handoff_permission": "allowed",
            "world_action_inputs": [
                "canonical_state_representation",
                "qualified_structured_actions",
                "existing_simulator_teacher_trajectories",
                "occupancy_exposure_programs",
            ],
            "blocking_reasons": [
                "full_objective_task_family_count_below_two",
                "full_objective_holdout_family_count_below_two",
                "headline_evaluation_capability_incomplete",
            ],
            "v2_development_runs": rescue_run_results,
            "provenance": {
                "v2_evidence_bundle": _artifact_ref(RESCUE_BUNDLE_PATH, repo_root),
                "objective_capability": _artifact_ref(
                    RESCUE_OBJECTIVE_CAPABILITY_PATH, repo_root
                ),
                "current_capability_matrix": _artifact_ref(
                    CAPABILITY_PATH, repo_root
                ),
            },
        }
        write_json(safe_repo_path(repo_root, ROUTE_CLOSEOUT_PATH), route_closeout)
        claim_scope = {
            "schema_version": "neuronsifter-c1-claim-scope-v1",
            "status": "foundation_transfer_claim_prohibited",
            "created_at": _now(),
            "g3_status": "blocked_incomplete",
            "empirical_result_status": "unavailable",
            "allowed_internal_wording": [
                "NeuronSifter reduced-core state encoder implementation",
                "public CNS transcriptomics development smoke",
                "C1 rescue audit closed without foundation transfer evidence",
            ],
            "prohibited_wording": [
                "foundation model established",
                "general-purpose CNS foundation model",
                "multimodal foundation model",
                "cross-modal prediction",
                "clinical generalization",
            ],
            "student_smoke_checkpoint_public_release_allowed": False,
            "c2_handoff_permission": "allowed",
            "claim_wording": "no foundation transfer claim",
            "provenance": _artifact_ref(ROUTE_CLOSEOUT_PATH, repo_root),
        }
        write_json(safe_repo_path(repo_root, CLAIM_SCOPE_PATH), claim_scope)
        closeout_artifacts = {
            "route_closeout": _artifact_ref(ROUTE_CLOSEOUT_PATH, repo_root),
            "claim_scope": _artifact_ref(CLAIM_SCOPE_PATH, repo_root),
        }
    decision = {
        "schema_version": "neuronsifter-c1-rescue-feasibility-decision-v1",
        "status": decision_status,
        "created_at": _now(),
        "selected_option_id": rescue["selected_option_id"] if decision_status == "rescue_feasible" else "none",
        "v1_corpus_uid": parent["corpus_uid"],
        "v1_corpus_identity_hash": parent["identity_hash"],
        "v1_mutation_performed": False,
        "v2_corpus_uid": bundle["corpus_uid"],
        "v2_corpus_identity_hash": bundle["corpus_hash"],
        "v2_evidence_status": validation["status"],
        "conditions": conditions,
        "data_reachable_task_families": reachable_tasks,
        "data_reachable_holdout_families": reachable_holdouts,
        "full_objective_reachable_task_families": full_objective_tasks,
        "full_objective_reachable_holdout_families": full_objective_holdouts,
        "headline_evaluation_capability": "incomplete",
        "g3_preflight_training_allowed": unified.get("training_allowed") is True,
        "headline_job_matrix_generation_allowed": False,
        "large_or_headline_submission_allowed": False,
        "foundation_claim_allowed": False,
        "student_smoke_checkpoint_release_allowed": False,
        "c2_handoff_permission": (
            "allowed"
            if decision_status == "pivot_to_c2_recommended"
            else "not_selected_while_rescue_feasible"
        ),
        "blockers": [
            {
                "code": "c1r_full_objective_task_coverage_below_minimum",
                "current": len(full_objective_tasks),
                "required": int(rescue["required_task_family_count"]),
            },
            {
                "code": "c1r_full_objective_holdout_coverage_below_minimum",
                "current": len(full_objective_holdouts),
                "required": int(rescue["required_holdout_family_count"]),
            },
            {
                "code": "c1r_headline_implementation_incomplete",
                "capabilities": missing_capabilities,
            }
        ] if decision_status == "pivot_to_c2_recommended" else [],
        "artifacts": {
            "current_capability_matrix": _artifact_ref(CAPABILITY_PATH, repo_root),
            "gap_decomposition": _artifact_ref(GAP_PATH, repo_root),
            "candidate_source_audit": _artifact_ref(CANDIDATE_PATH, repo_root),
            "candidate_independence_graph": _artifact_ref(INDEPENDENCE_PATH, repo_root),
            "v2_option_matrix": _artifact_ref(OPTION_PATH, repo_root),
            "v2_evidence_bundle": _artifact_ref(RESCUE_BUNDLE_PATH, repo_root),
            "v2_evidence_validation": _artifact_ref(RESCUE_VALIDATION_PATH, repo_root),
            "v2_objective_capability": _artifact_ref(
                RESCUE_OBJECTIVE_CAPABILITY_PATH, repo_root
            ),
            **closeout_artifacts,
        },
        "source_state": {
            "v1_preflight_status": primary_preflight["status"],
            "v1_inventory_status": inventory["status"],
            "v1_baseline_registry_status": baselines["status"],
            "v1_job_submission_status": job_matrix["submission_status"],
            "v1_g3_status": g3["g3_status"],
            "v1_gate_status": gate["status"],
        },
    }
    write_json(safe_repo_path(repo_root, DECISION_PATH), decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DECISION_PATH)
    args = parser.parse_args()
    decision = build(REPO_ROOT)
    if args.output != DECISION_PATH:
        write_json(safe_repo_path(REPO_ROOT, args.output), decision)
    print(json.dumps(decision, sort_keys=True))
    if decision["status"] not in {
        "rescue_feasible",
        "pivot_to_c2_recommended",
        "blocked_external_dependency",
    }:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
