"""Preregister and build an outcome-blind prospective development-data queue."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping

from jsonschema import Draft202012Validator
import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.scripts.builders.build_retrospective_public_cns_ranking_intake import (  # noqa: E402
    _binding,
    _canonical_digest,
)
from Libs.scripts.builders.build_retrospective_public_design_enriched_ranking_intake import (  # noqa: E402
    CNS_MODEL_TERMS,
    _named_intervention_entities,
)
from Libs.scripts.builders.build_retrospective_public_source_data_first_intake import (  # noqa: E402
    OUTCOME_BEARING_TERMS,
    _biostudies_inventory,
    _collect_exclusions as _collect_retrospective_exclusions,
    _contains_any,
    _europe_pmc_inventory,
    _local_inventory,
    _request_json,
)
from Libs.scripts.builders.build_prospective_protocol_scientific_review import (  # noqa: E402
    _payload_binding,
    _repo_path,
)
from Libs.utils.data.raw_data_registry import (  # noqa: E402
    load_config,
    write_json,
    write_text_atomic,
    write_tsv,
)


CONFIG_PATH = REPO_ROOT / "Libs/configs/config_stack.yaml"
CONFIG_SCHEMA_PATH = (
    REPO_ROOT
    / "Libs/configs/schema/prospective_development_data_intake_config.schema.json"
)
PREREGISTRATION_SCHEMA_PATH = (
    REPO_ROOT
    / "Libs/configs/schema/prospective_development_data_intake_preregistration.schema.json"
)
EXCLUSION_SCHEMA_PATH = (
    REPO_ROOT
    / "Libs/configs/schema/prospective_development_exclusion_snapshot.schema.json"
)
INTAKE_SCHEMA_PATH = (
    REPO_ROOT / "Libs/configs/schema/prospective_development_data_intake.schema.json"
)
IMPLEMENTATION_FREEZE_V1_SCHEMA_PATH = (
    REPO_ROOT
    / "Libs/configs/schema/prospective_development_intake_implementation_freeze.schema.json"
)
IMPLEMENTATION_FREEZE_V1_PATH = (
    REPO_ROOT
    / "Input/manifests/prospective_development/prospective_development_intake_implementation_freeze.json"
)
IMPLEMENTATION_FREEZE_V2_SCHEMA_PATH = (
    REPO_ROOT
    / "Libs/configs/schema/prospective_development_intake_implementation_freeze_v2.schema.json"
)
IMPLEMENTATION_FREEZE_V2_PATH = (
    REPO_ROOT
    / "Input/manifests/prospective_development/prospective_development_intake_implementation_freeze_v2.json"
)
IMPLEMENTATION_FREEZE_SCHEMA_PATH = (
    REPO_ROOT
    / "Libs/configs/schema/prospective_development_intake_implementation_freeze_v3.schema.json"
)
IMPLEMENTATION_FREEZE_PATH = (
    REPO_ROOT
    / "Input/manifests/prospective_development/prospective_development_intake_implementation_freeze_v3.json"
)
IMPLEMENTATION_FREEZE_INVALIDATION_SCHEMA_PATH = (
    REPO_ROOT
    / "Libs/configs/schema/prospective_development_intake_implementation_freeze_invalidation.schema.json"
)
IMPLEMENTATION_FREEZE_INVALIDATION_PATH = (
    REPO_ROOT
    / "Output/audits/prospective_development_intake_implementation_freeze_v1_invalidation.json"
)
IMPLEMENTATION_FREEZE_V2_INVALIDATION_SCHEMA_PATH = (
    REPO_ROOT
    / "Libs/configs/schema/prospective_development_intake_implementation_freeze_v2_invalidation.schema.json"
)
IMPLEMENTATION_FREEZE_V2_INVALIDATION_PATH = (
    REPO_ROOT
    / "Output/audits/prospective_development_intake_implementation_freeze_v2_invalidation.json"
)
ACCESS_BLOCKED_INVALIDATION_SCHEMA_PATH = (
    REPO_ROOT
    / "Libs/configs/schema/prospective_development_data_intake_invalidation.schema.json"
)
ACCESS_BLOCKED_INVALIDATION_PATH = (
    REPO_ROOT
    / "Output/audits/prospective_development_data_intake_access_blocked_invalidation.json"
)
ACCESS_BLOCKED_AUDIT_PATH = (
    REPO_ROOT / "Output/audits/prospective_development_data_intake_access_blocked.json"
)
ACCESS_BLOCKED_UNIVERSE_PATH = (
    REPO_ROOT
    / "Input/manifests/prospective_development/prospective_development_metadata_universe_access_blocked.tsv"
)
ACCESS_BLOCKED_QUEUE_PATH = (
    REPO_ROOT
    / "Input/manifests/prospective_development/prospective_development_source_queue_access_blocked.tsv"
)
ACCESS_BLOCKED_SNAPSHOT_PATH = (
    REPO_ROOT
    / "Input/raw/public/prospective_development/metadata_snapshot_access_blocked.json"
)

TERMINAL_INTAKE_STATUSES = frozenset(
    {"development_source_queue_frozen", "development_source_queue_exhausted"}
)
BUILDER_PATH = (
    REPO_ROOT / "Libs/scripts/builders/build_prospective_development_data_intake.py"
)
VALIDATOR_PATH = (
    REPO_ROOT
    / "Libs/scripts/validators/validate_prospective_development_data_intake.py"
)
TEST_PATH = REPO_ROOT / "Libs/tests/test_prospective_development_data_intake.py"
PREREGISTRATION_WRAPPER_PATH = (
    REPO_ROOT
    / "Libs/exp/evaluation/run_prospective_development_intake_preregistration.sbatch"
)
DISCOVERY_WRAPPER_PATH = (
    REPO_ROOT
    / "Libs/exp/evaluation/run_prospective_development_metadata_discovery.sbatch"
)

DEVELOPMENT_AUTHORITY_PATH = (
    REPO_ROOT / "Output/audits/prospective_development_data_authority.json"
)
RECONSTRUCTION_DECISION_PATH = (
    REPO_ROOT / "Output/audits/prospective_twin_reconstruction_decision.json"
)
TRAINING_AUTHORIZATION_PATH = (
    REPO_ROOT
    / "Input/manifests/retrospective_public/prospective_twin_training_authorization.json"
)
SEMANTIC_EXCLUSION_PATH = (
    REPO_ROOT / "Output/audits/six_context_semantic_exclusion_audit.json"
)
SIX_CONTEXT_MATRIX_PATH = (
    REPO_ROOT
    / "Input/manifests/retrospective_public/six_context_zero_shot_evaluation_matrix.tsv"
)
PUBLIC_BLOCKS_PATH = (
    REPO_ROOT
    / "Output/audits/public_foundation_corpus_rescue_canonical_observation_blocks.json"
)
PRIOR_METADATA_UNIVERSE_PATH = (
    REPO_ROOT / "Input/manifests/retrospective_public/source_data_metadata_universe.tsv"
)

UNIVERSE_FIELDS = (
    "asset_uid",
    "lineage_uid",
    "repository_family",
    "repository_record_uid",
    "study_title",
    "publication_uid",
    "doi",
    "release_date",
    "public_access_status",
    "asset_name",
    "asset_type",
    "asset_size_bytes",
    "asset_url",
    "local_path",
    "metadata_source",
    "metadata_text_sha256",
    "biological_system",
    "intervention_modality",
    "named_agents",
    "named_agent_count",
    "baseline_control_signal",
    "post_action_signal",
    "dose_signal",
    "timepoint_signal",
    "replication_signal",
    "functional_endpoint_class",
    "transcriptomic_auxiliary_signal",
    "empirical_signal",
    "machine_readable_signal",
    "role_capabilities",
    "six_context_exclusion_status",
    "eligibility_status",
    "rejection_reason",
    "source_asset_content_retrieved",
    "development_outcome_access",
    "six_context_outcome_access",
    "protected_holdout_access",
)

QUEUE_FIELDS = (
    "intake_uid",
    "queue_sequence",
    "preregistration_digest",
    "parent_state_digest",
    "metadata_universe_digest",
    "asset_uid",
    "lineage_uid",
    "repository_family",
    "repository_record_uid",
    "study_title",
    "publication_uid",
    "doi",
    "asset_name",
    "asset_type",
    "asset_size_bytes",
    "asset_url",
    "local_path",
    "biological_system",
    "intervention_modality",
    "named_agents",
    "baseline_control_signal",
    "post_action_signal",
    "dose_signal",
    "timepoint_signal",
    "replication_signal",
    "functional_endpoint_class",
    "transcriptomic_auxiliary_signal",
    "role_capabilities",
    "queue_priority",
    "queue_status",
    "source_asset_content_retrieved_before_priority",
    "development_outcomes_retrieved_before_priority",
    "six_context_outcome_access",
    "protected_holdout_access",
)

PROTEIN_TERMS = (
    "antibody",
    "protein delivery",
    "protein treatment",
    "recombinant protein",
    "enzyme replacement",
    "biologic",
    "peptide treatment",
    "growth factor",
    "cytokine treatment",
    "aav delivery",
)
COMPOUND_TERMS = (
    "drug",
    "compound",
    "small molecule",
    "inhibitor",
    "agonist",
    "antagonist",
    "pharmacological",
    "chemical treatment",
)
CONTROL_TERMS = ("control", "vehicle", "untreated", "baseline", "mock", "dmso")
POST_ACTION_TERMS = (
    "treated",
    "treatment",
    "post-treatment",
    "post treatment",
    "after treatment",
    "exposure",
    "stimulated",
    "perturbation",
)
DOSE_TERMS = ("dose", "dosage", "concentration", " nM", " uM", " µM", "mg/kg")
TIME_TERMS = ("time course", "time-course", "hour", "day", "week", " h ", " hr ")
REPLICATION_TERMS = (
    "biological replicate",
    "replicate",
    "donor",
    "individual data",
    "sdrf",
    "series matrix",
    "sample metadata",
)
FUNCTIONAL_ENDPOINTS = {
    "electrophysiology": ("electrophysiology", "voltage", "spike", "patch clamp"),
    "calcium_activity": ("calcium", "ca2+", "calcium imaging"),
    "cell_viability_survival": ("survival", "viability", "cell death", "apoptosis"),
    "neurite_morphology": ("neurite", "axon", "dendrite", "morphology"),
    "synaptic_phenotype": ("synaptic", "synapse", "puncta"),
    "behavior_motor": ("motor", "behavior", "locomotor"),
    "quantitative_pathology": ("aggregate", "pathology", "amyloid", "tau", "tdp-43"),
}
TRANSCRIPTOMIC_TERMS = (
    "rna-seq",
    "rna seq",
    "transcriptome",
    "transcriptomic",
    "gene expression",
    "expression matrix",
    "series matrix",
)
DESIGN_TABLE_TERMS = ("sdrf", "series matrix", "sample metadata", "phenodata")
SIMULATOR_TERMS = ("simulation", "simulator", "in silico", "predicted trajectory")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _schema_errors(payload: Mapping[str, Any], schema_path: Path) -> list[str]:
    schema = _read_json(schema_path)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: list(error.absolute_path),
    )
    return [f"{list(error.absolute_path)}: {error.message}" for error in errors]


def _load_policy() -> dict[str, Any]:
    config = load_config(CONFIG_PATH)
    policy = config.get("prospective_development_data_intake")
    if not isinstance(policy, dict):
        raise ValueError("Missing prospective_development_data_intake configuration")
    errors = _schema_errors(policy, CONFIG_SCHEMA_PATH)
    if errors:
        raise ValueError(
            "Development intake config schema errors: " + "; ".join(errors[:8])
        )
    authorization = policy["authorization"]
    if len(policy["repository_families"]) != int(
        authorization["repository_family_cap"]
    ):
        raise ValueError("Repository family count does not match the frozen cap")
    if sum(int(item["asset_cap"]) for item in policy["repository_families"]) > int(
        authorization["metadata_asset_cap"]
    ):
        raise ValueError("Per-family caps exceed the metadata asset cap")
    return policy


def _output_paths(policy: Mapping[str, Any]) -> dict[str, Path]:
    paths = {key: _repo_path(str(value)) for key, value in policy["outputs"].items()}
    return paths


def _assert_parent_state(policy: Mapping[str, Any]) -> dict[str, Any]:
    authority = _read_json(DEVELOPMENT_AUTHORITY_PATH)
    decision = _read_json(RECONSTRUCTION_DECISION_PATH)
    authorization = _read_json(TRAINING_AUTHORIZATION_PATH)
    semantic = _read_json(SEMANTIC_EXCLUSION_PATH)
    expected = policy["parent_state"]
    observed = {
        "development_domain_status": authority.get("status"),
        "method_preregistration_status": _read_json(
            REPO_ROOT / "Output/audits/prospective_method_preregistration.json"
        ).get("status"),
        "historical_checkpoint_status": _read_json(
            REPO_ROOT / "Output/audits/frozen_zero_shot_method_authority.json"
        ).get("status"),
        "training_authorized": authorization.get("training_authorized"),
        "gpu_authorized": authorization.get("gpu_authorized"),
        "six_context_outcome_access": authorization.get("six_context_outcome_access"),
        "protected_holdout_access": authorization.get("protected_holdout_access"),
    }
    if observed != expected:
        raise ValueError(f"Development intake parent state drifted: {observed}")
    if decision.get("status") != "NO_INDEPENDENT_PROSPECTIVE_TRAINING_DOMAIN":
        raise ValueError(
            "Development-domain decision is not the expected negative Gate"
        )
    if semantic.get("six_context_outcome_access") != 0:
        raise ValueError("Six-context outcome access is nonzero")
    return {
        "development_data_authority": _binding(DEVELOPMENT_AUTHORITY_PATH),
        "reconstruction_decision": _binding(RECONSTRUCTION_DECISION_PATH),
        "training_authorization": _binding(TRAINING_AUTHORIZATION_PATH),
        "semantic_exclusion": _binding(SEMANTIC_EXCLUSION_PATH),
    }


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [
            {key: str(value or "") for key, value in row.items()}
            for row in csv.DictReader(handle, delimiter="\t")
        ]


def _collect_exclusions() -> dict[str, Any]:
    six_rows = _read_tsv(SIX_CONTEXT_MATRIX_PATH)
    public_blocks = _read_json(PUBLIC_BLOCKS_PATH)
    retrospective = _collect_retrospective_exclusions()
    prior_universe = (
        _read_tsv(PRIOR_METADATA_UNIVERSE_PATH)
        if PRIOR_METADATA_UNIVERSE_PATH.is_file()
        else []
    )
    publication_uids = {
        value
        for row in six_rows
        for value in row.get("publication_uids", "").split("|")
        if value
    }
    action_uids = {
        value
        for row in six_rows
        for value in row.get("action_uids", "").split("|")
        if value
    }
    control_uids = {
        value
        for row in six_rows
        for value in row.get("control_uids", "").split("|")
        if value
    }
    source_accessions = {
        str(record.get("source_accession") or "")
        for record in public_blocks.get("qualified_blocks", [])
        if record.get("source_accession")
    }
    publication_groups = {
        str(record.get("publication_accession_group") or "")
        for record in public_blocks.get("qualified_blocks", [])
        if record.get("publication_accession_group")
    }
    repository_records = {
        row.get("repository_record_uid", "") for row in prior_universe
    } | set(retrospective["repository_record_uids"])
    return {
        "evaluation_context_uids": sorted(row["context_uid"] for row in six_rows),
        "six_context_publication_uids": sorted(publication_uids),
        "six_context_action_uids": sorted(action_uids),
        "six_context_control_uids": sorted(control_uids),
        "existing_development_source_accessions": sorted(source_accessions),
        "existing_development_publication_groups": sorted(publication_groups),
        "prior_repository_record_uids": sorted(
            value for value in repository_records if value
        ),
        "prior_publication_uids": retrospective["publication_uids"],
        "prior_dois": retrospective["dois"],
        "prior_source_artifact_paths": retrospective["source_artifact_paths"],
    }


def _write_role_requirements(path: Path, policy: Mapping[str, Any]) -> None:
    content = yaml.safe_dump(
        {"prospective_development_role_requirements": policy["role_requirements"]},
        sort_keys=False,
        allow_unicode=True,
    )
    write_text_atomic(path, content)


def _exclusion_snapshot(
    policy: Mapping[str, Any], exclusions: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_version": "neuronsifter-prospective-development-exclusion-snapshot-v1",
        "status": "frozen_before_development_metadata_access",
        "intake_uid": policy["intake_uid"],
        "created_at": _now(),
        "six_context_binding": _binding(SIX_CONTEXT_MATRIX_PATH),
        "evaluation_context_uids": exclusions["evaluation_context_uids"],
        "publication_uids": exclusions["six_context_publication_uids"],
        "action_uids": exclusions["six_context_action_uids"],
        "control_uids": exclusions["six_context_control_uids"],
        "development_source_exclusions": {
            "source_accessions": exclusions["existing_development_source_accessions"],
            "publication_groups": exclusions["existing_development_publication_groups"],
            "prior_repository_record_uids": exclusions["prior_repository_record_uids"],
        },
        "six_context_outcome_access": 0,
        "protected_holdout_access": 0,
    }
    errors = _schema_errors(payload, EXCLUSION_SCHEMA_PATH)
    if errors:
        raise ValueError(
            "Development exclusion snapshot errors: " + "; ".join(errors[:8])
        )
    return payload


def _validate_preregistration(
    payload: Mapping[str, Any], policy: Mapping[str, Any]
) -> None:
    errors = _schema_errors(payload, PREREGISTRATION_SCHEMA_PATH)
    if errors:
        raise ValueError("Development preregistration errors: " + "; ".join(errors[:8]))
    digest_payload = dict(payload)
    observed = str(digest_payload.pop("preregistration_digest"))
    if _canonical_digest(digest_payload) != observed:
        raise ValueError("Development preregistration digest drifted")
    for field in (
        "intake_uid",
        "authorization",
        "parent_state",
        "repository_families",
        "role_requirements",
        "metadata_policy",
        "selection_policy",
        "boundaries",
        "outputs",
    ):
        if payload[field] != policy[field]:
            raise ValueError(f"Development preregistration {field} drifted")


def preregister(*, write_output: bool = True) -> dict[str, Any]:
    policy = _load_policy()
    paths = _output_paths(policy)
    if paths["preregistration_manifest"].is_file():
        payload = _read_json(paths["preregistration_manifest"])
        _validate_preregistration(payload, policy)
        return payload
    parent_bindings = _assert_parent_state(policy)
    exclusions = _collect_exclusions()
    exclusion_snapshot = _exclusion_snapshot(policy, exclusions)
    payload: dict[str, Any] = {
        "schema_version": "neuronsifter-prospective-development-data-intake-preregistration-v1",
        "status": "preregistered_before_new_development_outcome_access",
        "intake_uid": policy["intake_uid"],
        "preregistered_at": _now(),
        "authorization": policy["authorization"],
        "parent_state": policy["parent_state"],
        "parent_state_digest": _canonical_digest(policy["parent_state"]),
        "parent_bindings": parent_bindings,
        "repository_families": policy["repository_families"],
        "role_requirements": policy["role_requirements"],
        "metadata_policy": policy["metadata_policy"],
        "selection_policy": policy["selection_policy"],
        "exclusions": exclusions,
        "boundaries": policy["boundaries"],
        "outputs": policy["outputs"],
    }
    payload["preregistration_digest"] = _canonical_digest(payload)
    _validate_preregistration(payload, policy)
    if write_output:
        write_json(paths["preregistration_manifest"], payload)
        _write_role_requirements(paths["role_requirements_snapshot"], policy)
        write_json(paths["exclusion_snapshot"], exclusion_snapshot)
    return payload


def _validate_implementation_freeze(
    payload: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    *,
    verify_current_implementation: bool = True,
) -> None:
    errors = _schema_errors(payload, IMPLEMENTATION_FREEZE_SCHEMA_PATH)
    if errors:
        raise ValueError(
            "Development implementation freeze errors: " + "; ".join(errors[:8])
        )
    digest_payload = dict(payload)
    observed = str(digest_payload.pop("freeze_digest"))
    if _canonical_digest(digest_payload) != observed:
        raise ValueError("Development implementation freeze digest drifted")
    if payload["preregistration_digest"] != preregistration["preregistration_digest"]:
        raise ValueError("Implementation freeze preregistration binding drifted")
    for field in (
        "preregistration_binding",
        "superseded_freeze_binding",
        "invalidation_binding",
    ):
        binding = payload[field]
        path = _repo_path(str(binding["path"]))
        if _binding(path) != binding:
            raise ValueError(f"Implementation freeze lineage binding drifted: {field}")
    if verify_current_implementation:
        for binding in payload["implementation_bindings"].values():
            path = _repo_path(str(binding["path"]))
            if _binding(path) != binding:
                raise ValueError(f"Implementation binding drifted: {binding['path']}")


def freeze_implementation(*, write_output: bool = True) -> dict[str, Any]:
    policy = _load_policy()
    paths = _output_paths(policy)
    preregistration = preregister(write_output=write_output)
    if IMPLEMENTATION_FREEZE_PATH.is_file():
        payload = _read_json(IMPLEMENTATION_FREEZE_PATH)
        terminal_intake_exists = False
        if paths["intake_audit"].is_file():
            terminal_intake_exists = (
                _read_json(paths["intake_audit"]).get("status")
                in TERMINAL_INTAKE_STATUSES
            )
        _validate_implementation_freeze(
            payload,
            preregistration,
            verify_current_implementation=not terminal_intake_exists,
        )
        return payload
    if not IMPLEMENTATION_FREEZE_V2_PATH.is_file():
        raise FileNotFoundError(
            "The preserved v2 implementation freeze is required before creating v3"
        )
    if not ACCESS_BLOCKED_INVALIDATION_PATH.is_file():
        raise FileNotFoundError(
            "The access-blocked intake attempt must be archived before creating v3"
        )
    invalidation = (
        _read_json(IMPLEMENTATION_FREEZE_V2_INVALIDATION_PATH)
        if IMPLEMENTATION_FREEZE_V2_INVALIDATION_PATH.is_file()
        else {
            "schema_version": "neuronsifter-prospective-development-intake-implementation-freeze-v2-invalidation-v1",
            "status": "invalidated_after_access_blocked_metadata_attempt",
            "reason": "inherited_loopback_proxy_unavailable_on_cpu03",
            "created_at": _now(),
            "invalidated_freeze_binding": _binding(IMPLEMENTATION_FREEZE_V2_PATH),
            "access_blocked_intake_binding": _binding(ACCESS_BLOCKED_AUDIT_PATH),
            "metadata_requests_attempted": True,
            "metadata_assets_retrieved": 0,
            "source_asset_content_retrieved": False,
            "development_outcome_access": 0,
            "six_context_outcome_access": 0,
            "protected_holdout_access": 0,
        }
    )
    invalidation_errors = _schema_errors(
        invalidation, IMPLEMENTATION_FREEZE_V2_INVALIDATION_SCHEMA_PATH
    )
    if invalidation_errors:
        raise ValueError(
            "Development implementation invalidation errors: "
            + "; ".join(invalidation_errors[:8])
        )
    if write_output and not IMPLEMENTATION_FREEZE_V2_INVALIDATION_PATH.is_file():
        write_json(IMPLEMENTATION_FREEZE_V2_INVALIDATION_PATH, invalidation)
    invalidation_binding = (
        _binding(IMPLEMENTATION_FREEZE_V2_INVALIDATION_PATH)
        if IMPLEMENTATION_FREEZE_V2_INVALIDATION_PATH.is_file()
        else _payload_binding(IMPLEMENTATION_FREEZE_V2_INVALIDATION_PATH, invalidation)
    )
    payload: dict[str, Any] = {
        "schema_version": "neuronsifter-prospective-development-intake-implementation-freeze-v3",
        "status": "frozen_before_repository_metadata_retry",
        "intake_uid": policy["intake_uid"],
        "created_at": _now(),
        "preregistration_digest": preregistration["preregistration_digest"],
        "preregistration_binding": _binding(paths["preregistration_manifest"]),
        "implementation_bindings": {
            "config": _binding(
                REPO_ROOT / "Libs/configs/data/prospective_development_data_intake.yaml"
            ),
            "builder": _binding(BUILDER_PATH),
            "validator": _binding(VALIDATOR_PATH),
            "tests": _binding(TEST_PATH),
            "preregistration_wrapper": _binding(PREREGISTRATION_WRAPPER_PATH),
            "discovery_wrapper": _binding(DISCOVERY_WRAPPER_PATH),
            "config_schema": _binding(CONFIG_SCHEMA_PATH),
            "preregistration_schema": _binding(PREREGISTRATION_SCHEMA_PATH),
            "exclusion_schema": _binding(EXCLUSION_SCHEMA_PATH),
            "intake_schema": _binding(INTAKE_SCHEMA_PATH),
            "validation_schema": _binding(
                REPO_ROOT
                / "Libs/configs/schema/prospective_development_data_intake_validation.schema.json"
            ),
            "implementation_freeze_schema": _binding(IMPLEMENTATION_FREEZE_SCHEMA_PATH),
            "implementation_freeze_v2_schema": _binding(
                IMPLEMENTATION_FREEZE_V2_SCHEMA_PATH
            ),
            "implementation_freeze_v2_invalidation_schema": _binding(
                IMPLEMENTATION_FREEZE_V2_INVALIDATION_SCHEMA_PATH
            ),
            "access_blocked_invalidation_schema": _binding(
                ACCESS_BLOCKED_INVALIDATION_SCHEMA_PATH
            ),
        },
        "superseded_freeze_binding": _binding(IMPLEMENTATION_FREEZE_V2_PATH),
        "invalidation_binding": invalidation_binding,
        "repair_reason": "direct_network_retry_after_inherited_loopback_proxy_failure",
        "source_asset_content_retrieved": False,
        "development_outcome_access": 0,
        "six_context_outcome_access": 0,
        "protected_holdout_access": 0,
    }
    payload["freeze_digest"] = _canonical_digest(payload)
    _validate_implementation_freeze(payload, preregistration)
    if write_output:
        write_json(IMPLEMENTATION_FREEZE_PATH, payload)
    return payload


def _geo_inventory(
    session: requests.Session, family: Mapping[str, Any]
) -> tuple[str, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    statuses: list[str] = []
    seen: set[str] = set()
    for query in family["query_plan"]:
        status, payload = _request_json(
            session,
            str(family["search_url"]),
            params={
                "db": "gds",
                "term": query,
                "retmode": "json",
                "retmax": min(32, int(family["asset_cap"])),
            },
        )
        statuses.append(status)
        if status != "resolved" or payload is None:
            continue
        ids = [
            str(value) for value in payload.get("esearchresult", {}).get("idlist", [])
        ]
        for start in range(0, len(ids), 20):
            batch = ids[start : start + 20]
            summary_status, summary = _request_json(
                session,
                str(family["summary_url"]),
                params={"db": "gds", "id": ",".join(batch), "retmode": "json"},
                max_bytes=8 * 1024 * 1024,
            )
            statuses.append(summary_status)
            if summary_status != "resolved" or summary is None:
                continue
            result = summary.get("result", {})
            for uid in result.get("uids", []):
                if len(rows) >= int(family["asset_cap"]):
                    break
                record = result.get(str(uid), {})
                if not isinstance(record, dict):
                    continue
                accession = str(record.get("accession") or "").strip()
                if not accession or accession in seen:
                    continue
                seen.add(accession)
                title = " ".join(str(record.get("title") or "").split())
                sample_titles = []
                for sample in record.get("samples", [])[:24]:
                    if isinstance(sample, dict):
                        sample_titles.append(
                            " ".join(str(sample.get("title") or "").split())
                        )
                pmids = record.get("pubmedids") or []
                pmid = str(pmids[0]) if isinstance(pmids, list) and pmids else ""
                metadata_text = " ".join(
                    (
                        title,
                        str(record.get("gdstype") or ""),
                        str(record.get("taxon") or ""),
                        " ".join(sample_titles),
                    )
                )
                rows.append(
                    {
                        "repository_family": family["family_uid"],
                        "repository_record_uid": accession,
                        "asset_name": f"{accession}_series_matrix_or_raw_bundle.zip",
                        "asset_type": "supplementary_bundle",
                        "asset_size_bytes": 0,
                        "asset_url": str(record.get("ftplink") or ""),
                        "local_path": "",
                        "study_title": title,
                        "publication_uid": f"pubmed:{pmid}" if pmid else "",
                        "doi": "",
                        "release_date": str(record.get("pdat") or ""),
                        "public_access_status": "public",
                        "metadata_source": "ncbi_gds_esearch_esummary_metadata",
                        "metadata_text": metadata_text,
                        "metadata_design_text": metadata_text,
                    }
                )
    overall = (
        "resolved"
        if statuses and all(value == "resolved" for value in statuses)
        else "access_blocked"
    )
    return overall, rows


def _sanitize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    title = " ".join(str(record.get("study_title") or "").split())
    asset_name = str(record.get("asset_name") or "")
    metadata_design = str(record.get("metadata_design_text") or "")
    if not metadata_design and record.get("metadata_source") in {
        "ncbi_gds_esearch_esummary_metadata",
        "local_path_and_file_stat_only",
        "test_metadata",
    }:
        metadata_design = str(record.get("metadata_text") or "")
    return {
        **record,
        "metadata_text": " ".join(
            (
                title,
                asset_name,
                str(record.get("repository_record_uid") or ""),
                metadata_design,
            )
        )[:8000],
    }


def _biological_system(text: str) -> str:
    lowered = text.lower()
    for label, terms in (
        ("neural_organoid", ("organoid",)),
        ("ipsc_neuron", ("ipsc", "induced pluripotent")),
        ("primary_neuron", ("primary neuron",)),
        ("microglia", ("microglia",)),
        ("astrocyte", ("astrocyte",)),
        ("neuronal_system", ("neuron", "neuronal", "neural")),
        ("brain_tissue", ("brain", "cns")),
    ):
        if _contains_any(lowered, terms):
            return label
    return ""


def _intervention_modality(text: str) -> str:
    lowered = text.lower()
    treatment_context = _contains_any(
        lowered,
        ("treatment", "treated", "delivery", "exposure", "dose", "perturbation"),
    )
    if treatment_context and _contains_any(lowered, PROTEIN_TERMS):
        return "protein_biologic_delivery"
    if _contains_any(lowered, COMPOUND_TERMS):
        return "small_molecule_compound"
    return ""


def _endpoint_class(text: str) -> str:
    lowered = text.lower()
    for label, terms in FUNCTIONAL_ENDPOINTS.items():
        if _contains_any(lowered, terms):
            return label
    return ""


def _signal(text: str, terms: Iterable[str], pattern: str | None = None) -> bool:
    lowered = text.lower()
    return _contains_any(lowered, terms) or bool(
        pattern and re.search(pattern, text, re.I)
    )


def _lineage_uid(record: Mapping[str, Any]) -> str:
    return (
        "development-lineage:"
        + _canonical_digest(
            {
                "repository_family": record["repository_family"],
                "repository_record_uid": record["repository_record_uid"],
                "publication_uid": record.get("publication_uid", ""),
                "doi": str(record.get("doi") or "").lower(),
            }
        )[:24]
    )


def _asset_uid(record: Mapping[str, Any]) -> str:
    return (
        "development-asset:"
        + _canonical_digest(
            {
                "repository_family": record["repository_family"],
                "repository_record_uid": record["repository_record_uid"],
                "asset_name": record["asset_name"],
                "asset_url": record.get("asset_url", ""),
                "local_path": record.get("local_path", ""),
            }
        )[:24]
    )


def _assess_asset(
    record: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    exclusions: Mapping[str, Any],
) -> dict[str, str]:
    sanitized = _sanitize_record(record)
    text = str(sanitized["metadata_text"])
    lowered = text.lower()
    title = str(sanitized.get("study_title") or "")
    modality = _intervention_modality(text)
    system = _biological_system(text)
    agents = _named_intervention_entities(text)
    baseline = _signal(text, CONTROL_TERMS)
    post_action = _signal(text, POST_ACTION_TERMS)
    design_table = _signal(text, DESIGN_TABLE_TERMS)
    dose = (
        _signal(
            text,
            DOSE_TERMS,
            r"\b\d+(?:\.\d+)?\s*(?:nm|um|µm|mm|mg/kg|ng/ml|ug/ml|µg/ml)\b",
        )
        or design_table
    )
    timepoint = (
        _signal(
            text,
            TIME_TERMS,
            r"\b\d+(?:\.\d+)?\s*(?:h|hr|hrs|hour|hours|d|day|days|week|weeks)\b",
        )
        or design_table
    )
    replication = _signal(text, REPLICATION_TERMS) or design_table
    endpoint = _endpoint_class(text)
    transcriptomic = _signal(text, TRANSCRIPTOMIC_TERMS)
    machine = str(sanitized.get("asset_type") or "").lower() in {
        value.lstrip(".")
        for value in policy["metadata_policy"]["machine_readable_extensions"]
    } | {"supplementary_bundle"}
    empirical = not _contains_any(lowered, SIMULATOR_TERMS)
    publication_uid = str(sanitized.get("publication_uid") or "")
    doi = str(sanitized.get("doi") or "").lower()
    repository_record_uid = str(sanitized.get("repository_record_uid") or "")
    exclusion_status = "passed"
    rejection_reason = ""
    if publication_uid in exclusions["six_context_publication_uids"]:
        exclusion_status = "six_context_exact_publication"
        rejection_reason = exclusion_status
    elif repository_record_uid in exclusions["existing_development_source_accessions"]:
        exclusion_status = "existing_development_lineage"
        rejection_reason = exclusion_status
    elif repository_record_uid in exclusions["prior_repository_record_uids"]:
        exclusion_status = "prior_repository_record"
        rejection_reason = exclusion_status
    elif doi and doi in set(exclusions["prior_dois"]):
        exclusion_status = "prior_doi_lineage"
        rejection_reason = exclusion_status
    elif any(
        doi.startswith(prefix)
        for prefix in policy["selection_policy"]["excluded_doi_prefixes"]
    ):
        rejection_reason = "source_policy_excluded"
    elif str(sanitized.get("public_access_status") or "") not in {
        "public",
        "public_open_access",
        "public_local_snapshot",
    }:
        rejection_reason = "not_public"
    elif _contains_any(title.lower(), OUTCOME_BEARING_TERMS):
        rejection_reason = "outcome_bearing_study_title"
    elif not publication_uid and not repository_record_uid:
        rejection_reason = "missing_study_identity"
    elif not system or not _contains_any(lowered, CNS_MODEL_TERMS):
        rejection_reason = "no_cns_or_neural_model"
    elif not modality:
        rejection_reason = "no_target_intervention_modality"
    elif not agents:
        rejection_reason = "no_named_agent_metadata"
    elif not baseline:
        rejection_reason = "no_baseline_or_control_metadata"
    elif not post_action:
        rejection_reason = "no_post_action_metadata"
    elif not dose:
        rejection_reason = "no_dose_or_intensity_metadata"
    elif not timepoint:
        rejection_reason = "no_timepoint_metadata"
    elif not replication:
        rejection_reason = "no_replication_metadata"
    elif not machine:
        rejection_reason = "no_machine_readable_source_asset"
    elif not empirical:
        rejection_reason = "not_empirical"
    elif not endpoint and not transcriptomic:
        rejection_reason = "no_functional_or_auxiliary_endpoint"
    capabilities = ["TRAIN_CANDIDATE"]
    if endpoint:
        capabilities.extend(
            [
                "MODEL_SELECTION_VALIDATION_CANDIDATE",
                "NON_EVAL_QUALIFICATION_CANDIDATE",
            ]
        )
    return {
        "asset_uid": _asset_uid(sanitized),
        "lineage_uid": _lineage_uid(sanitized),
        "repository_family": str(sanitized["repository_family"]),
        "repository_record_uid": repository_record_uid,
        "study_title": title,
        "publication_uid": publication_uid,
        "doi": doi,
        "release_date": str(sanitized.get("release_date") or ""),
        "public_access_status": str(sanitized.get("public_access_status") or ""),
        "asset_name": str(sanitized.get("asset_name") or ""),
        "asset_type": str(sanitized.get("asset_type") or ""),
        "asset_size_bytes": str(sanitized.get("asset_size_bytes") or 0),
        "asset_url": str(sanitized.get("asset_url") or ""),
        "local_path": str(sanitized.get("local_path") or ""),
        "metadata_source": str(sanitized.get("metadata_source") or ""),
        "metadata_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "biological_system": system,
        "intervention_modality": modality,
        "named_agents": ";".join(agents),
        "named_agent_count": str(len(agents)),
        "baseline_control_signal": str(baseline).lower(),
        "post_action_signal": str(post_action).lower(),
        "dose_signal": str(dose).lower(),
        "timepoint_signal": str(timepoint).lower(),
        "replication_signal": str(replication).lower(),
        "functional_endpoint_class": endpoint,
        "transcriptomic_auxiliary_signal": str(transcriptomic).lower(),
        "empirical_signal": str(empirical).lower(),
        "machine_readable_signal": str(machine).lower(),
        "role_capabilities": ";".join(capabilities),
        "six_context_exclusion_status": exclusion_status,
        "eligibility_status": "eligible" if not rejection_reason else "rejected",
        "rejection_reason": rejection_reason,
        "source_asset_content_retrieved": "false",
        "development_outcome_access": "0",
        "six_context_outcome_access": "0",
        "protected_holdout_access": "0",
    }


def _priority_key(
    record: Mapping[str, str], selected: list[Mapping[str, str]]
) -> tuple[object, ...]:
    selected_families = {row["repository_family"] for row in selected}
    selected_modalities = {row["intervention_modality"] for row in selected}
    capabilities = record["role_capabilities"].split(";")
    return (
        record["intervention_modality"] != "protein_biologic_delivery",
        not bool(record["functional_endpoint_class"]),
        record["intervention_modality"] in selected_modalities,
        -len(capabilities),
        record["repository_family"] in selected_families,
        record["asset_type"] == "supplementary_bundle",
        record["lineage_uid"],
        record["asset_uid"],
    )


def _queue_rows(
    eligible: list[dict[str, str]],
    *,
    preregistration: Mapping[str, Any],
    universe_digest: str,
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    remaining = list(eligible)
    used_lineages: set[str] = set()
    cap = int(preregistration["authorization"]["preregistered_source_lineage_cap"])
    while remaining and len(selected) < cap:
        remaining.sort(key=lambda record: _priority_key(record, selected))
        candidate = remaining.pop(0)
        if candidate["lineage_uid"] in used_lineages:
            continue
        used_lineages.add(candidate["lineage_uid"])
        selected.append(candidate)
    rows = []
    for index, record in enumerate(selected, start=1):
        rows.append(
            {
                "intake_uid": str(preregistration["intake_uid"]),
                "queue_sequence": str(index),
                "preregistration_digest": str(
                    preregistration["preregistration_digest"]
                ),
                "parent_state_digest": str(preregistration["parent_state_digest"]),
                "metadata_universe_digest": universe_digest,
                **{field: record[field] for field in QUEUE_FIELDS if field in record},
                "queue_priority": str(index),
                "queue_status": "preregistered_for_development_source_recovery",
                "source_asset_content_retrieved_before_priority": "false",
                "development_outcomes_retrieved_before_priority": "false",
                "six_context_outcome_access": "0",
                "protected_holdout_access": "0",
            }
        )
    return rows


def _universe_digest(
    rows: list[dict[str, str]], preregistration: Mapping[str, Any]
) -> str:
    return _canonical_digest(
        {
            "contract": "prospective_development_metadata_universe_v1",
            "intake_uid": preregistration["intake_uid"],
            "preregistration_digest": preregistration["preregistration_digest"],
            "records": rows,
        }
    )


def _queue_digest(
    rows: list[dict[str, str]], preregistration: Mapping[str, Any]
) -> str:
    return _canonical_digest(
        {
            "contract": "prospective_development_source_queue_v1",
            "intake_uid": preregistration["intake_uid"],
            "preregistration_digest": preregistration["preregistration_digest"],
            "records": rows,
        }
    )


def archive_access_blocked(*, write_output: bool = True) -> dict[str, Any]:
    policy = _load_policy()
    paths = _output_paths(policy)
    preregistration = preregister(write_output=write_output)
    if ACCESS_BLOCKED_INVALIDATION_PATH.is_file():
        payload = _read_json(ACCESS_BLOCKED_INVALIDATION_PATH)
        errors = _schema_errors(payload, ACCESS_BLOCKED_INVALIDATION_SCHEMA_PATH)
        if errors:
            raise ValueError(
                "Access-block invalidation errors: " + "; ".join(errors[:8])
            )
        return payload
    audit = _read_json(paths["intake_audit"])
    if audit.get("status") != "repository_metadata_access_blocked":
        raise ValueError("Only an access-blocked intake can be archived for retry")
    if audit.get("metadata_asset_count") != 0 or audit.get("queue_lineage_count") != 0:
        raise ValueError("Access-blocked retry requires zero retrieved metadata assets")
    universe_text = paths["metadata_universe"].read_text(encoding="utf-8")
    queue_text = paths["source_queue"].read_text(encoding="utf-8")
    snapshot = _read_json(paths["metadata_snapshot"])
    if write_output:
        write_text_atomic(ACCESS_BLOCKED_UNIVERSE_PATH, universe_text)
        write_text_atomic(ACCESS_BLOCKED_QUEUE_PATH, queue_text)
        write_json(ACCESS_BLOCKED_SNAPSHOT_PATH, snapshot)
        write_json(ACCESS_BLOCKED_AUDIT_PATH, audit)
    archived_bindings = {
        "metadata_universe": _binding(ACCESS_BLOCKED_UNIVERSE_PATH),
        "source_queue": _binding(ACCESS_BLOCKED_QUEUE_PATH),
        "metadata_snapshot": _binding(ACCESS_BLOCKED_SNAPSHOT_PATH),
        "intake_audit": _binding(ACCESS_BLOCKED_AUDIT_PATH),
    }
    payload = {
        "schema_version": "neuronsifter-prospective-development-data-intake-invalidation-v1",
        "status": "access_blocked_attempt_archived_for_same_query_retry",
        "reason": "inherited_loopback_proxy_unavailable_on_cpu03",
        "created_at": _now(),
        "preregistration_digest": preregistration["preregistration_digest"],
        "archived_artifact_bindings": archived_bindings,
        "retry_scope": "same_preregistered_queries_and_caps_only",
        "source_asset_content_retrieved": False,
        "development_outcome_access": 0,
        "six_context_outcome_access": 0,
        "protected_holdout_access": 0,
    }
    errors = _schema_errors(payload, ACCESS_BLOCKED_INVALIDATION_SCHEMA_PATH)
    if errors:
        raise ValueError("Access-block invalidation errors: " + "; ".join(errors[:8]))
    if write_output:
        write_json(ACCESS_BLOCKED_INVALIDATION_PATH, payload)
    return payload


def _assert_access_blocked_retry(
    audit: Mapping[str, Any], preregistration: Mapping[str, Any]
) -> None:
    if audit.get("status") in TERMINAL_INTAKE_STATUSES:
        raise ValueError(
            "Development intake is terminally frozen; an access-blocked retry would "
            "constitute an unauthorized second intake"
        )
    if audit.get("status") != "repository_metadata_access_blocked":
        raise ValueError(
            "Access-blocked retry requires the current canonical audit to remain "
            "repository_metadata_access_blocked"
        )
    if audit.get("metadata_asset_count") != 0 or audit.get("queue_lineage_count") != 0:
        raise ValueError("Access-blocked retry requires zero retrieved metadata assets")
    if not ACCESS_BLOCKED_INVALIDATION_PATH.is_file():
        raise ValueError(
            "Access-blocked retry requires the archived invalidation artifact"
        )
    invalidation = _read_json(ACCESS_BLOCKED_INVALIDATION_PATH)
    errors = _schema_errors(invalidation, ACCESS_BLOCKED_INVALIDATION_SCHEMA_PATH)
    if errors:
        raise ValueError("Access-block invalidation errors: " + "; ".join(errors[:8]))
    if invalidation.get("preregistration_digest") != preregistration.get(
        "preregistration_digest"
    ):
        raise ValueError("Access-block invalidation preregistration binding drifted")
    for binding in invalidation["archived_artifact_bindings"].values():
        path = _repo_path(str(binding["path"]))
        if _binding(path) != binding:
            raise ValueError(
                f"Archived access-block binding drifted: {binding['path']}"
            )
    if _read_json(ACCESS_BLOCKED_AUDIT_PATH) != dict(audit):
        raise ValueError(
            "Canonical access-blocked audit drifted from its archived copy"
        )


def discover(
    *, write_output: bool = True, retry_access_blocked: bool = False
) -> dict[str, Any]:
    policy = _load_policy()
    paths = _output_paths(policy)
    preregistration = preregister(write_output=write_output)
    existing_audit = (
        _read_json(paths["intake_audit"]) if paths["intake_audit"].is_file() else None
    )
    if existing_audit is not None:
        if not retry_access_blocked:
            return existing_audit
        _assert_access_blocked_retry(existing_audit, preregistration)
    elif retry_access_blocked:
        raise ValueError(
            "Access-blocked retry requires the current canonical access-blocked audit"
        )
    freeze_implementation(write_output=write_output)
    if not retry_access_blocked and any(
        paths[key].exists()
        for key in ("metadata_universe", "source_queue", "metadata_snapshot")
    ):
        raise ValueError(
            "Partial development intake outputs exist without canonical audit"
        )
    _assert_parent_state(policy)
    session = requests.Session()
    session.trust_env = False
    session.headers.update({"User-Agent": "NeuronSifter-development-metadata/1.0"})
    raw_assets: list[dict[str, Any]] = []
    family_status: list[dict[str, Any]] = []
    local_exclusions = {
        "source_artifact_paths": preregistration["exclusions"][
            "prior_source_artifact_paths"
        ]
    }
    for family in policy["repository_families"]:
        if family["adapter"] == "bounded_local_file_metadata_inventory":
            status, records = _local_inventory(
                family,
                local_exclusions,
                maximum_single_asset_bytes=int(
                    policy["metadata_policy"]["maximum_single_asset_bytes"]
                ),
            )
        elif family["adapter"] == "biostudies_search_and_detail_metadata":
            status, records = _biostudies_inventory(session, dict(family))
        elif family["adapter"] == "europe_pmc_lite_supplement_metadata":
            status, records = _europe_pmc_inventory(session, dict(family))
        elif family["adapter"] == "ncbi_gds_esearch_esummary_metadata":
            status, records = _geo_inventory(session, family)
        else:
            raise ValueError(
                f"Unsupported development repository adapter: {family['adapter']}"
            )
        records = [_sanitize_record(record) for record in records]
        family_status.append(
            {
                "family_uid": family["family_uid"],
                "adapter": family["adapter"],
                "status": status,
                "metadata_asset_count": len(records),
                "asset_cap": int(family["asset_cap"]),
            }
        )
        raw_assets.extend(records)
    deduplicated: dict[str, dict[str, Any]] = {}
    for record in raw_assets:
        deduplicated.setdefault(_asset_uid(record), record)
    if len(deduplicated) > int(policy["authorization"]["metadata_asset_cap"]):
        raise ValueError("Development metadata universe exceeded the preregistered cap")
    universe_rows = [
        _assess_asset(record, policy=policy, exclusions=preregistration["exclusions"])
        for _, record in sorted(deduplicated.items())
    ]
    eligible = [row for row in universe_rows if row["eligibility_status"] == "eligible"]
    universe_digest = _universe_digest(universe_rows, preregistration)
    queue_rows = _queue_rows(
        eligible,
        preregistration=preregistration,
        universe_digest=universe_digest,
    )
    queue_digest = _queue_digest(queue_rows, preregistration)
    access_blocked = any(record["status"] != "resolved" for record in family_status)
    status = (
        "repository_metadata_access_blocked"
        if access_blocked
        else "development_source_queue_frozen"
        if queue_rows
        else "development_source_queue_exhausted"
    )
    next_action = (
        "retry_same_preregistered_metadata_requests"
        if access_blocked
        else "recover_frozen_development_sources"
        if queue_rows
        else "stop_bounded_intake_exhausted"
    )
    snapshot = {
        "schema_version": "neuronsifter-prospective-development-metadata-snapshot-v1",
        "retrieved_at": _now(),
        "retrieval_mode": "repository_metadata_only",
        "preregistration_binding": _binding(paths["preregistration_manifest"]),
        "repository_family_status": family_status,
        "metadata_asset_count": len(universe_rows),
        "metadata_eligible_count": len(eligible),
        "queue_lineage_count": len(queue_rows),
        "metadata_universe_digest": universe_digest,
        "queue_digest": queue_digest,
        "source_boundaries": {
            "source_asset_content_retrieved": False,
            "development_numeric_outcomes_retrieved": False,
            "six_context_outcome_access": 0,
            "protected_holdout_access": 0,
        },
    }
    serialized = (
        json.dumps(snapshot, indent=2, ensure_ascii=True).encode("utf-8") + b"\n"
    )
    snapshot_binding = {
        "path": paths["metadata_snapshot"].relative_to(REPO_ROOT).as_posix(),
        "bytes": len(serialized),
        "sha256": hashlib.sha256(serialized).hexdigest(),
    }
    capability_counts = Counter(
        capability
        for row in eligible
        for capability in row["role_capabilities"].split(";")
        if capability
    )
    modality_counts = Counter(row["intervention_modality"] for row in eligible)
    audit = {
        "schema_version": "neuronsifter-prospective-development-data-intake-v1",
        "status": status,
        "intake_uid": preregistration["intake_uid"],
        "preregistration_digest": preregistration["preregistration_digest"],
        "parent_state_digest": preregistration["parent_state_digest"],
        "repository_family_count": len(policy["repository_families"]),
        "metadata_asset_cap": int(policy["authorization"]["metadata_asset_cap"]),
        "metadata_asset_count": len(universe_rows),
        "metadata_eligible_count": len(eligible),
        "queue_lineage_count": len(queue_rows),
        "metadata_universe_digest": universe_digest,
        "queue_digest": queue_digest,
        "metadata_universe_path": paths["metadata_universe"]
        .relative_to(REPO_ROOT)
        .as_posix(),
        "queue_path": paths["source_queue"].relative_to(REPO_ROOT).as_posix(),
        "metadata_snapshot_binding": snapshot_binding,
        "repository_family_status": family_status,
        "capability_funnel": {
            "repository_assets_screened": len(universe_rows),
            "metadata_eligible": len(eligible),
            "independent_lineages_queued": len(queue_rows),
            "role_capability_counts": dict(sorted(capability_counts.items())),
            "modality_counts": dict(sorted(modality_counts.items())),
            "source_payloads_recovered": 0,
            "empirical_action_response_units": 0,
            "formal_roles_assigned": 0,
        },
        "queue_frozen_before_new_development_outcome_access": True,
        "development_numeric_outcomes_retrieved_before_queue_freeze": False,
        "six_context_outcome_access": 0,
        "protected_holdout_access": 0,
        "training_authorized": False,
        "gpu_authorized": False,
        "next_action": next_action,
        "queue_records": queue_rows,
        "created_at": _now(),
    }
    errors = _schema_errors(audit, INTAKE_SCHEMA_PATH)
    if errors:
        raise ValueError("Development intake schema errors: " + "; ".join(errors[:8]))
    if write_output:
        write_tsv(paths["metadata_universe"], list(UNIVERSE_FIELDS), universe_rows)
        write_tsv(paths["source_queue"], list(QUEUE_FIELDS), queue_rows)
        write_json(paths["metadata_snapshot"], snapshot)
        write_json(paths["intake_audit"], audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("preregister", "freeze", "archive-access-blocked", "discover")
    )
    parser.add_argument("--retry-access-blocked", action="store_true")
    args = parser.parse_args()
    if args.mode == "preregister":
        result = preregister(write_output=True)
    elif args.mode == "freeze":
        result = freeze_implementation(write_output=True)
    elif args.mode == "archive-access-blocked":
        result = archive_access_blocked(write_output=True)
    else:
        result = discover(
            write_output=True, retry_access_blocked=args.retry_access_blocked
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
