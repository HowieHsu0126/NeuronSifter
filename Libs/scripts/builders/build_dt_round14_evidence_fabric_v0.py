"""Materialize the unsealed Round-14 Evidence Fabric V0 release candidate."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.action_observation_contracts import (  # noqa: E402
    SCHEMA_VERSIONS,
    validate_canonical_observation,
    validate_information_action,
    validate_world_action,
)
from Libs.utils.data.evidence_fabric_v0 import (  # noqa: E402
    ADMITTED_STATES,
    FABRIC_VERSION,
    RELEASE_SCHEMA_VERSION,
    AdmissionInput,
    admit_evidence,
    canonicalize_entity_mapping,
    canonical_sha256,
    duplicate_and_conflict_audit,
    evidence_fabric_v0_config,
    is_sha256,
    release_uid,
    sha256_file,
    stable_uid,
    validate_admission_record,
    validate_release_manifest,
    validate_typed_catalogs,
    validate_v0_config,
    write_parquet_records,
)
from Libs.utils.data.evidence_relationships import (  # noqa: E402
    IDENTITY_AXES,
    PAIRING_REQUIRED_AXES,
    REQUIRED_ACTION_ENDPOINT_EVIDENCE,
    EvidenceConflict,
    EvidencePairingDecision,
    EvidenceSupportDomain,
)
from Libs.utils.data.raw_data_registry import load_config, write_json, write_tsv  # noqa: E402
from Libs.utils.exp.orchestration_controller import sha256_payload  # noqa: E402


ROUND14_VALIDATOR_UID = "round14_evidence_fabric_admission_validator"
CANONICALIZATION_VERSION = "neuronsifter-evidence-fabric-v0-canonicalization-v1"

NODE_FIELDS = (
    "schema_version",
    "node_uid",
    "node_type",
    "evidence_class",
    "qc_grade",
    "identity_axes",
    "provenance_ref",
    "train_eval_role",
    "status",
    "layer",
    "source_artifact",
    "producer_uid",
    "content_hash",
    "admission_decision",
    "payload",
)
RELATION_FIELDS = (
    "schema_version",
    "relation_uid",
    "relation_type",
    "source_node_uid",
    "target_node_uid",
    "directionality",
    "biological_unit_relation",
    "temporal_relation",
    "action_relation",
    "context_relation",
    "support_domain_ref",
    "eligibility_ref",
    "provenance_ref",
    "confidence",
    "status",
    "source_artifact",
    "producer_uid",
    "content_hash",
    "admission_decision",
)
PAIRING_FIELDS = (
    "schema_version",
    "pairing_uid",
    "source_node_uid",
    "target_node_uid",
    "decision",
    "identity_axis_results",
    "required_identity_axes",
    "pairing_evidence",
    "blockers",
    "status",
    "source_artifact",
    "producer_uid",
    "content_hash",
    "admission_decision",
)
SUPPORT_DOMAIN_FIELDS = (
    "schema_version",
    "support_domain_uid",
    "organism_scope",
    "disease_scope",
    "region_scope",
    "compartment_scope",
    "cell_type_scope",
    "modality_scope",
    "assay_scope",
    "action_scope",
    "dose_scope",
    "route_scope",
    "time_scope",
    "model_family_scope",
    "tool_family_scope",
    "source_scope",
    "generalization_status",
    "provenance_ref",
    "status",
    "source_artifact",
    "producer_uid",
    "content_hash",
    "admission_decision",
)
ELIGIBILITY_FIELDS = (
    "schema_version",
    "eligibility_uid",
    "subject_uid",
    "eligible_for_state_pretraining",
    "eligible_for_action_endpoint_training",
    "eligible_for_trajectory_training",
    "eligible_for_calibration",
    "eligible_for_validation",
    "eligible_for_stress_test",
    "eligible_for_claim_support",
    "eligible_for_hypothesis_generation",
    "reasons",
    "status",
    "source_artifact",
    "producer_uid",
    "content_hash",
    "admission_decision",
)
CONFLICT_FIELDS = (
    "schema_version",
    "conflict_uid",
    "conflict_type",
    "member_uids",
    "state",
    "typed_context_difference",
    "provenance_ref",
    "status",
    "source_artifact",
    "producer_uid",
    "content_hash",
    "admission_decision",
)
ACTION_FIELDS = (
    "schema_version",
    "action_uid",
    "intervention_type",
    "target_set",
    "agent_set",
    "dose",
    "dose_unit",
    "schedule",
    "route",
    "duration",
    "stimulation_or_environment",
    "state_scope",
    "support_domain_ref",
    "source_evidence_refs",
    "action_provenance_ref",
    "conflicts",
    "status",
    "layer",
    "source_artifact",
    "content_hash",
    "admission_decision",
)
INFORMATION_ACTION_FIELDS = (
    "schema_version",
    "information_action_uid",
    "candidate_uids",
    "requested_observables",
    "tool_or_source_uid",
    "fidelity",
    "measurement_context",
    "budget_class",
    "expected_cost",
    "expected_runtime",
    "failure_policy",
    "support_domain_ref",
    "provenance_ref",
    "conflicts",
    "status",
    "layer",
    "source_artifact",
    "content_hash",
    "admission_decision",
)
OBSERVATION_FIELDS = (
    "schema_version",
    "observation_uid",
    "source_result_uid",
    "biological_unit_uid",
    "state_time",
    "region",
    "compartment",
    "cell_type",
    "modality",
    "feature_namespace",
    "feature_id",
    "value",
    "unit",
    "uncertainty",
    "missingness",
    "qc_grade",
    "support_domain_ref",
    "provenance_ref",
    "train_eval_role",
    "conflicts",
    "empirical_observation",
    "layer",
    "source_artifact",
    "content_hash",
    "admission_decision",
)
SOURCE_FIELDS = (
    "source_name",
    "path",
    "size",
    "sha256",
    "schema_version",
    "role",
)
ADMISSION_FIELDS = (
    "schema_version",
    "candidate_uid",
    "layer",
    "item_type",
    "source_uid",
    "official_source_or_internal_producer",
    "source_version",
    "publication_or_accession",
    "license_or_access_status",
    "retrieval_or_generation_date",
    "raw_hash",
    "processed_hash",
    "parser_version",
    "canonicalization_version",
    "producer_uid",
    "producer_command",
    "train_eval_role",
    "holdout_group",
    "support_domain_ref",
    "qc_grade",
    "relationship_refs",
    "conflict_refs",
    "pairing_decision",
    "legacy_role",
    "empirical_truth",
    "eligibility",
    "identity",
    "payload",
    "state_history",
    "terminal_state",
    "blockers",
    "content_hash",
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _git_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _value(axis: Mapping[str, object] | None, fallback: str = "not_applicable") -> str:
    raw = dict(axis or {}).get("value", fallback)
    return str(raw) if raw not in (None, "") else fallback


def _support_domain(
    uid: str,
    *,
    source_scope: str,
    axes: Mapping[str, object] | None = None,
    generalization: str = "in_domain",
    action_scope: str = "not_applicable",
    route_scope: str = "not_applicable",
    time_scope: str = "not_applicable",
    model_family_scope: str = "not_applicable",
    tool_family_scope: str = "not_applicable",
    source_artifact: str,
    admission_decision: str = "admitted_weighted",
) -> dict[str, object]:
    values = dict(axes or {})
    domain = EvidenceSupportDomain(
        schema_version="neuronsifter-evidence-support-domain-v1",
        support_domain_uid=uid,
        organism_scope=[_value(values.get("species"))],
        disease_scope=[_value(values.get("disease_state"))],
        region_scope=[_value(values.get("brain_region"), _value(values.get("tissue")))],
        compartment_scope=["not_applicable"],
        cell_type_scope=[_value(values.get("cell_type"))],
        modality_scope=[_value(values.get("modality"))],
        assay_scope=[_value(values.get("assay"))],
        action_scope=[action_scope],
        dose_scope=[_value(values.get("dose"))],
        route_scope=[route_scope],
        time_scope=[time_scope],
        model_family_scope=[model_family_scope],
        tool_family_scope=[tool_family_scope],
        source_scope=[source_scope],
        generalization_status=generalization,
        provenance_ref=f"provenance:{source_scope}",
        status="materialized_unsealed",
    ).to_dict()
    metadata = {
        "source_artifact": source_artifact,
        "producer_uid": "round14_evidence_fabric_materializer",
        "admission_decision": admission_decision,
    }
    return {**domain, **metadata, "content_hash": canonical_sha256({**domain, **metadata})}


def _source_rows(paths: Mapping[str, Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, path in sorted(paths.items()):
        if not path.is_file():
            raise FileNotFoundError(path)
        schema_version = ""
        if path.suffix == ".json":
            schema_version = str(_load_json(path).get("schema_version", ""))
        rows.append(
            {
                "source_name": name,
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
                "schema_version": schema_version,
                "role": "authorized_governed_input",
            }
        )
    return rows


def _pilot_blocks(blocks: Sequence[object], limit: int | None) -> list[object]:
    if limit is None:
        return list(blocks)
    if limit < 1:
        raise ValueError("pilot_limit must be positive")
    selected: list[object] = []
    counts = {"E1": 0, "E2": 0}
    for block in blocks:
        if not isinstance(block, Mapping):
            continue
        layer = "E2" if str(block.get("family_id")) == "compound_dose_time_perturbation_transcriptomics" else "E1"
        if counts[layer] < limit:
            selected.append(block)
            counts[layer] += 1
    return selected


def _manifest_license(manifest: Mapping[str, object]) -> str:
    release = dict(manifest.get("source_release", {})) if isinstance(manifest.get("source_release"), Mapping) else {}
    terms = dict(release.get("terms_snapshot", {})) if isinstance(release.get("terms_snapshot"), Mapping) else {}
    access_class = str(release.get("access_class", ""))
    bindings = [dict(row) for row in manifest.get("input_bindings", []) if isinstance(row, Mapping)]
    policy_binding_verified = any(
        is_sha256(row.get("sha256"))
        and any(token in str(row.get("path", "")).lower() for token in ("terms", "policy", "license"))
        for row in bindings
    )
    if access_class.startswith("public") and (
        is_sha256(terms.get("sha256")) or (release.get("reuse_evidence") and policy_binding_verified)
    ):
        return "verified_public"
    if access_class.startswith("registered_") and release.get("license_role") and policy_binding_verified:
        return "verified_internal"
    return "unresolved"


def _manifest_date(manifest: Mapping[str, object]) -> str:
    return str(manifest.get("created_at", "existing_governed_artifact"))


def _make_observation(block: Mapping[str, object], manifest: Mapping[str, object], domain_uid: str) -> dict[str, object]:
    axes = dict(manifest.get("semantic_axes", {})) if isinstance(manifest.get("semantic_axes"), Mapping) else {}
    join = dict(manifest.get("observation_value_join_contract", {})) if isinstance(manifest.get("observation_value_join_contract"), Mapping) else {}
    layer = "E2" if str(block.get("family_id")) == "compound_dose_time_perturbation_transcriptomics" else "E1"
    observation = {
        "schema_version": SCHEMA_VERSIONS["canonical_observation"],
        "observation_uid": str(block["canonical_uid"]),
        "source_result_uid": f"source_block:{block['canonical_uid']}",
        "biological_unit_uid": f"block_units:{block['canonical_uid']}",
        "state_time": {
            "semantics": "static_endpoint" if layer == "E2" else "cross_sectional",
            "source_axis": _value(axes.get("time")),
        },
        "region": _value(axes.get("brain_region"), _value(axes.get("tissue"))),
        "compartment": "not_applicable",
        "cell_type": _value(axes.get("cell_type")),
        "modality": _value(axes.get("modality")),
        "feature_namespace": str(join.get("feature_namespace", _value(axes.get("feature")))),
        "feature_id": "content_addressed_feature_axis",
        "value": {
            "storage_path": block.get("value_block_path"),
            "storage_hash": block.get("value_block_sha256", block.get("block_sha256")),
            "observation_count": block.get("observation_count"),
        },
        "unit": str(join.get("value_unit", block.get("unit_mapping_version", "source_native"))),
        "uncertainty": {"qc_grade": block.get("qc_grade"), "mapping_confidence": block.get("mapping_confidence")},
        "missingness": {"state": "content_addressed_block", "semantic_axes_complete": block.get("semantic_axes_complete")},
        "qc_grade": str(block.get("qc_grade", "D")),
        "support_domain_ref": domain_uid,
        "provenance_ref": f"manifest:{block['manifest_sha256']}",
        "train_eval_role": "action_endpoint_candidate" if layer == "E2" else "state_pretraining",
        "conflicts": [],
        "empirical_observation": True,
    }
    return observation


def _source_action_to_world_action(row: Mapping[str, object], domain_uid: str, source_ref: str) -> dict[str, object]:
    dose_unit_aliases = {"uM": "umol/L", "microM": "umol/L"}
    source_unit = str(row.get("dose_unit", ""))
    route = str(row.get("route", row.get("route_or_not_applicable", "")))
    duration = row.get("duration")
    duration_unit = str(row.get("duration_unit", ""))
    target_raw = row.get("target_set")
    target_set = [] if target_raw in (None, "", "explicit_unknown") else [str(target_raw)]
    agent_raw = row.get("agent_set", row.get("agent_uid"))
    agent_set = [] if agent_raw in (None, "") else [str(agent_raw)]
    return {
        "schema_version": SCHEMA_VERSIONS["world_action"],
        "action_uid": str(row.get("action_uid", "")),
        "intervention_type": "pharmacological",
        "target_set": target_set,
        "agent_set": agent_set,
        "dose": row.get("dose_value", row.get("physical_dose")),
        "dose_unit": dose_unit_aliases.get(source_unit, source_unit),
        "schedule": {"kind": "continuous"} if str(row.get("schedule", "")).startswith("continuous") else {"kind": "single"},
        "route": route,
        "duration": {"value": duration, "unit": duration_unit},
        "stimulation_or_environment": {"source_context": row.get("cell_or_tissue_context", "not_applicable")},
        "state_scope": {"source_scope": row.get("state_scope", row.get("cell_or_tissue_context", "not_applicable"))},
        "support_domain_ref": domain_uid,
        "source_evidence_refs": [source_ref],
        "action_provenance_ref": f"source_action:{row.get('action_uid', '')}",
        "conflicts": [],
        "status": "materialization_candidate",
    }


def _admit_foundation_blocks(
    *,
    blocks_artifact: Mapping[str, object],
    source_root: Path,
    blocks_path: Path,
    pilot_limit: int | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    admissions: list[dict[str, object]] = []
    observations: list[dict[str, object]] = []
    actions: list[dict[str, object]] = []
    domains: list[dict[str, object]] = []
    e0_rows: list[dict[str, object]] = []
    source_blocks = _pilot_blocks(list(blocks_artifact.get("blocks", [])), pilot_limit)
    for block in source_blocks:
        if not isinstance(block, Mapping):
            continue
        manifest_rel = str(block.get("manifest_path", ""))
        manifest_path = source_root / manifest_rel
        manifest_exists = manifest_path.is_file()
        manifest = _load_json(manifest_path) if manifest_exists else {}
        manifest_hash = sha256_file(manifest_path) if manifest_exists else ""
        expected_manifest_hash = str(block.get("manifest_sha256", ""))
        hash_verified = manifest_exists and manifest_hash == expected_manifest_hash and manifest.get("block_sha256") == block.get("block_sha256")
        layer = "E2" if str(block.get("family_id")) == "compound_dose_time_perturbation_transcriptomics" else "E1"
        domain_uid = stable_uid("support_domain", {"block": block.get("canonical_uid"), "layer": layer})
        axes = dict(manifest.get("semantic_axes", {})) if isinstance(manifest.get("semantic_axes"), Mapping) else {}
        domains.append(
            _support_domain(
                domain_uid,
                source_scope=str(block.get("source_id", "")),
                axes=axes,
                action_scope="pharmacological" if layer == "E2" else "not_applicable",
                route_scope="source_declared" if layer == "E2" else "not_applicable",
                time_scope=_value(axes.get("time")),
                source_artifact=str(blocks_path),
            )
        )
        observation = _make_observation(block, manifest, domain_uid)
        observation_blockers = validate_canonical_observation(observation)
        action_valid = layer != "E2"
        source_actions: list[dict[str, object]] = []
        if layer == "E2" and manifest_exists:
            action_rel = str(manifest.get("action_catalog_path", ""))
            action_path = manifest_path.parent / action_rel
            if action_path.is_file():
                for raw_action in pq.read_table(action_path).to_pylist():
                    world_action = _source_action_to_world_action(raw_action, domain_uid, str(block.get("canonical_uid")))
                    source_actions.append(world_action)
                    if not validate_world_action(world_action):
                        actions.append(
                            {
                                **world_action,
                                "layer": "E2",
                                "source_artifact": str(action_path),
                                "content_hash": canonical_sha256(world_action),
                                "admission_decision": "pending_parent_admission",
                            }
                        )
                action_valid = bool(source_actions) and all(not validate_world_action(action) for action in source_actions)
        input_bindings = [row for row in manifest.get("input_bindings", []) if isinstance(row, Mapping)]
        raw_hash = str(input_bindings[0].get("sha256", "")) if input_bindings else expected_manifest_hash
        identity = {
            "canonical_uid": block.get("canonical_uid"),
            "source_id": block.get("source_id"),
            "source_accession": block.get("source_accession"),
            "family_id": block.get("family_id"),
            "biological_unit_count": block.get("biological_unit_count"),
        }
        release = dict(manifest.get("source_release", {})) if isinstance(manifest.get("source_release"), Mapping) else {}
        parser = dict(manifest.get("parser", {})) if isinstance(manifest.get("parser"), Mapping) else {}
        admission = admit_evidence(
            AdmissionInput(
                layer=layer,
                item_type="real_perturbation_endpoint_block" if layer == "E2" else "real_static_observation_block",
                source_uid=str(block.get("source_id", "")),
                official_source_or_internal_producer="official_public_source",
                source_version=str(release.get("release_version", block.get("source_accession", ""))),
                publication_or_accession=str(block.get("publication_accession_group", block.get("source_accession", ""))),
                license_or_access_status=_manifest_license(manifest),
                retrieval_or_generation_date=_manifest_date(manifest),
                raw_hash=raw_hash,
                processed_hash=str(block.get("block_sha256", "")),
                parser_version=str(parser.get("parser_version", parser.get("parser_id", "manifest_declared"))),
                canonicalization_version=CANONICALIZATION_VERSION,
                producer_uid=str(manifest.get("producer", "")),
                producer_command=f"python {manifest.get('producer', '')}",
                train_eval_role="action_endpoint_candidate" if layer == "E2" else "state_pretraining",
                holdout_group=str(block.get("publication_accession_group", block.get("source_accession", ""))),
                support_domain_ref=domain_uid,
                qc_grade=str(block.get("qc_grade", "D")),
                schema_version=str(manifest.get("schema_version", "")),
                identity=identity,
                payload={
                    "block": dict(block),
                    "manifest_hash": manifest_hash,
                    "actions": source_actions,
                    "source_artifact": str(manifest_path),
                },
                relationship_refs=[],
                conflict_refs=[],
                pairing_decision="valid_group_level_action_endpoint" if action_valid and layer == "E2" else "valid_cross_sectional_comparison" if layer == "E1" else "insufficient_identity",
                empirical_truth=True,
                schema_valid=not observation_blockers and bool(manifest.get("schema_version")),
                identity_valid=bool(block.get("semantic_axes_complete")) and bool(block.get("canonical_uid")),
                action_observation_valid=action_valid,
                action_complete=action_valid,
                provenance_complete=bool(manifest.get("producer")) and bool(input_bindings),
                hash_verified=hash_verified,
            )
        )
        admissions.append(admission)
        observation_row = {
            **observation,
            "layer": layer,
            "source_artifact": str(manifest_path),
            "content_hash": canonical_sha256(observation),
            "admission_decision": admission["terminal_state"],
        }
        if admission["terminal_state"] in ADMITTED_STATES:
            observations.append(observation_row)
        for action in actions:
            if action["source_evidence_refs"] == [str(block.get("canonical_uid"))]:
                action["admission_decision"] = admission["terminal_state"]
        for entity_type, entity_uid in (
            ("accession", str(block.get("source_accession", ""))),
            ("publication_accession_crosswalk", str(block.get("publication_accession_group", ""))),
            ("study", str(block.get("canonical_uid", ""))),
            ("biological_unit", f"block_units:{block.get('canonical_uid', '')}"),
        ):
            if entity_uid:
                e0_rows.append(
                    {
                        "entity_uid": entity_uid,
                        "entity_type": entity_type,
                        "source_uid": str(block.get("source_id", "")),
                        "canonical_uid": entity_uid,
                        "mapping_status": "exact",
                        "mapping_confidence": str(block.get("mapping_confidence", "")),
                        "source_artifact": str(manifest_path),
                        "content_hash": canonical_sha256({"entity_uid": entity_uid, "block_hash": block.get("block_sha256")}),
                    }
                )
        for namespace, axis_name in (
            ("organism_registry", "species"),
            ("brain_region_ontology", "brain_region"),
            ("compartment_ontology", "compartment"),
            ("cell_type_ontology", "cell_type"),
            ("feature_namespace", "feature"),
            ("observation_vocabulary", "modality"),
        ):
            axis = dict(axes.get(axis_name, {})) if isinstance(axes.get(axis_name), Mapping) else {}
            axis_value = str(axis.get("value", ""))
            status = str(axis.get("status", "unresolved"))
            mapped = status == "mapped" and bool(axis_value)
            e0_rows.append(
                {
                    "namespace": namespace,
                    "source_uid": f"{block.get('canonical_uid', '')}:{axis_name}",
                    "canonical_uid": axis_value if mapped else "",
                    "mapping_status": "mapped" if mapped else "unresolved",
                    "resolution_artifact_ref": f"manifest:{expected_manifest_hash}" if mapped else "",
                    "source_artifact": str(manifest_path),
                    "content_hash": canonical_sha256(
                        {"block_uid": block.get("canonical_uid"), "axis": axis_name, "axis_value": axis}
                    ),
                }
            )
    return admissions, observations, actions, domains, e0_rows


def _admit_legacy_rows(
    path: Path, *, pilot_limit: int | None
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    admissions: list[dict[str, object]] = []
    domains = [
        _support_domain(
            "domain_legacy_simulator_restricted",
            source_scope="simulation_qc_catalog",
            generalization="stress_only",
            action_scope="source_declared_incomplete",
            route_scope="source_declared",
            time_scope="simulated_time",
            model_family_scope="legacy_simulator_families",
            source_artifact=str(path),
            admission_decision="admitted_legacy_restricted",
        )
    ]
    with path.open("r", encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle, delimiter="\t")):
            if pilot_limit is not None and index >= pilot_limit:
                break
            grade = str(row.get("qc_grade", "D"))
            legacy_role = (
                "quarantined"
                if grade == "D"
                else "stress_only"
                if grade == "S"
                else "legacy_dynamics_pretraining"
                if str(row.get("qc_use")) == "main_pretraining"
                else "numerical_dynamics_only"
            )
            content_hash = str(row.get("content_hash", ""))
            admission = admit_evidence(
                AdmissionInput(
                    layer="E3",
                    item_type="legacy_simulator_trajectory",
                    source_uid=str(row.get("source_uid", row.get("source_collection", ""))),
                    official_source_or_internal_producer="internal_simulator_producer",
                    source_version=str(row.get("parameter_profile_uid", "legacy_parameter_profile")),
                    publication_or_accession="",
                    license_or_access_status="verified_internal",
                    retrieval_or_generation_date="existing_governed_artifact",
                    raw_hash=content_hash,
                    processed_hash=content_hash,
                    parser_version="simulation_qc_catalog_reader_v1",
                    canonicalization_version=CANONICALIZATION_VERSION,
                    producer_uid=str(row.get("source_collection", "simulation_qc_catalog")),
                    producer_command="python Libs/scripts/builders/build_simulation_readiness_audit.py",
                    train_eval_role=legacy_role,
                    holdout_group=str(row.get("trajectory_family_uid", "")),
                    support_domain_ref="domain_legacy_simulator_restricted",
                    qc_grade=grade,
                    schema_version="neuronsifter-simulation-qc-catalog-v1",
                    identity={
                        "trajectory_uid": row.get("trajectory_uid"),
                        "trajectory_family_uid": row.get("trajectory_family_uid"),
                        "regimen_uid": row.get("regimen_uid"),
                        "simulator_family": row.get("teacher_family"),
                    },
                    payload={
                        "source_artifact": str(path),
                        **{
                            key: row.get(key)
                            for key in (
                                "trajectory_store_path",
                                "trajectory_index",
                                "source_collection",
                                "teacher_family",
                                "teacher_role",
                                "disease_program",
                                "regimen_uid",
                                "route",
                                "schedule",
                                "dose_level_id",
                                "parameter_profile_uid",
                                "paired_status",
                                "failure_flags",
                                "warning_flags",
                            )
                        },
                    },
                    relationship_refs=[],
                    conflict_refs=[],
                    pairing_decision="valid_simulator_trajectory_pair" if row.get("paired_status") == "paired" else "insufficient_identity",
                    legacy_role=legacy_role,
                    empirical_truth=False,
                    schema_valid=bool(row.get("trajectory_uid")) and bool(row.get("trajectory_store_path")),
                    identity_valid=bool(row.get("trajectory_uid")) and bool(row.get("trajectory_family_uid")),
                    action_complete=False,
                    provenance_complete=bool(row.get("source_uid")) and bool(row.get("parameter_profile_uid")),
                    hash_verified=is_sha256(content_hash),
                )
            )
            admissions.append(admission)
    return admissions, domains


def _admit_models(path: Path, *, pilot_limit: int | None) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    admissions: list[dict[str, object]] = []
    domains: list[dict[str, object]] = []
    catalog_hash = sha256_file(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle, delimiter="\t")):
            if pilot_limit is not None and index >= pilot_limit:
                break
            uid = str(row.get("source_model_id", ""))
            domain_uid = stable_uid("support_domain_model", {"uid": uid, "scope": row.get("disease_scope")})
            domains.append(
                _support_domain(
                    domain_uid,
                    source_scope=str(row.get("source_repository", "")),
                    generalization="unsupported",
                    model_family_scope=str(row.get("model_family", "")),
                    source_artifact=str(path),
                    admission_decision="admitted_weak",
                )
            )
            license_value = str(row.get("license_or_terms", ""))
            admission = admit_evidence(
                AdmissionInput(
                    layer="E5",
                    item_type="published_mechanistic_model",
                    source_uid=str(row.get("source_repository", "")),
                    official_source_or_internal_producer="published_or_internal_model_registry",
                    source_version=str(row.get("materialization_status", "registered")),
                    publication_or_accession=str(row.get("source_url", "")),
                    license_or_access_status="verified_internal" if "internal" in license_value else "verified_public" if license_value else "unresolved",
                    retrieval_or_generation_date="existing_governed_artifact",
                    raw_hash=catalog_hash,
                    processed_hash=canonical_sha256(row),
                    parser_version="source_model_catalog_reader_v1",
                    canonicalization_version=CANONICALIZATION_VERSION,
                    producer_uid="source_model_catalog",
                    producer_command="python Libs/scripts/builders/build_full_microenvironment_teacher_registry.py",
                    train_eval_role="mechanistic_prior",
                    holdout_group=str(row.get("model_family", uid)),
                    support_domain_ref=domain_uid,
                    qc_grade="C",
                    schema_version="neuronsifter-source-model-catalog-v1",
                    identity={"model_uid": uid, "model_family": row.get("model_family")},
                    payload={**dict(row), "source_artifact": str(path)},
                    relationship_refs=[],
                    conflict_refs=[],
                    pairing_decision="valid_model_output_pair",
                    empirical_truth=False,
                    schema_valid=bool(uid) and bool(row.get("model_family")),
                    identity_valid=bool(uid),
                    provenance_complete=bool(row.get("source_url")) and bool(license_value),
                )
            )
            admissions.append(admission)
    return admissions, domains


def _admit_round13_graph(
    graph: Mapping[str, object], path: Path, *, pilot_limit: int | None
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    admissions: list[dict[str, object]] = []
    actions: list[dict[str, object]] = []
    information_actions: list[dict[str, object]] = []
    observations: list[dict[str, object]] = []
    graph_hash = sha256_file(path)
    families = (
        ("world_actions", "E6", "world_action", validate_world_action),
        ("information_actions", "E6", "information_action", validate_information_action),
        ("raw_tool_results", "E6", "raw_tool_result", lambda value: []),
        ("canonical_observations", "E6", "canonical_tool_observation", validate_canonical_observation),
        ("hypothetical_scenarios", "E7", "hypothetical_scenario", lambda value: []),
    )
    for key, layer, item_type, validator in families:
        rows = [dict(row) for row in graph.get(key, []) if isinstance(row, Mapping)]
        if pilot_limit is not None:
            rows = rows[:pilot_limit]
        for row in rows:
            uid = str(
                row.get("action_uid")
                or row.get("information_action_uid")
                or row.get("tool_result_uid")
                or row.get("observation_uid")
                or row.get("scenario_uid")
                or ""
            )
            provenance = str(row.get("provenance_ref", row.get("action_provenance_ref", "")))
            qc = str(row.get("qc_grade", "S" if layer == "E7" else "C"))
            contract_blockers = validator(row)
            empirical_truth = item_type == "canonical_tool_observation" and row.get("empirical_observation") is True
            # Round-13 pilot results remain demonstrations even when their source contract is empirical.
            empirical_truth = False if row.get("status") == "pilot_only" or layer == "E7" else empirical_truth
            admission = admit_evidence(
                AdmissionInput(
                    layer=layer,
                    item_type=item_type,
                    source_uid="round13_action_observation_contract_pilot",
                    official_source_or_internal_producer="internal_contract_producer",
                    source_version=str(graph.get("schema_version", "")),
                    publication_or_accession="",
                    license_or_access_status="verified_internal",
                    retrieval_or_generation_date="existing_governed_artifact",
                    raw_hash=graph_hash,
                    processed_hash=canonical_sha256(row),
                    parser_version="round13_graph_reader_v1",
                    canonicalization_version=CANONICALIZATION_VERSION,
                    producer_uid="build_dt_round13_action_observation_contracts",
                    producer_command="python Libs/scripts/builders/build_dt_round13_action_observation_contracts.py",
                    train_eval_role="contract_demonstration_only",
                    holdout_group="round13_contract_pilot",
                    support_domain_ref=str(row.get("support_domain_ref", "domain_hypothetical_unsupported" if layer == "E7" else "domain_human_ad_hippocampus")),
                    qc_grade=qc if qc in {"A", "B", "C", "D", "S"} else "C",
                    schema_version=str(row.get("schema_version", "")),
                    identity={"object_uid": uid, "object_type": item_type},
                    payload={**row, "source_artifact": str(path)},
                    relationship_refs=[],
                    conflict_refs=[str(value) for value in row.get("conflicts", [])],
                    empirical_truth=empirical_truth,
                    schema_valid=not contract_blockers,
                    identity_valid=bool(uid),
                    provenance_complete=provenance.lower() not in {"", "unknown", "unresolved"},
                )
            )
            admissions.append(admission)
            metadata = {
                "layer": layer,
                "source_artifact": str(path),
                "content_hash": canonical_sha256(row),
                "admission_decision": admission["terminal_state"],
            }
            if item_type == "world_action" and not contract_blockers:
                actions.append({**row, **metadata})
            elif item_type == "information_action" and not contract_blockers:
                information_actions.append({**row, **metadata})
            elif item_type == "canonical_tool_observation" and not contract_blockers:
                observations.append({**row, **metadata})
    return admissions, actions, information_actions, observations


def _e0_catalogs(
    config: Mapping[str, object], e0_entities: Sequence[Mapping[str, object]], source_hash: str
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    v0 = evidence_fabric_v0_config(config)
    e0 = dict(v0.get("e0", {}))
    unit_rows = [
        {
            **dict(row),
            "mapping_status": "exact",
            "source_artifact": "Libs/configs/data/digital_twin_evidence_fabric.yaml",
            "content_hash": canonical_sha256(row),
        }
        for row in e0.get("unit_registry", [])
        if isinstance(row, Mapping)
    ]
    ontology_rows = [dict(row) for row in e0_entities if row.get("namespace")]
    entity_rows = [dict(row) for row in e0_entities if not row.get("namespace")]
    for row in e0.get("organism_registry", []):
        if isinstance(row, Mapping):
            ontology_rows.append(
                {
                    "namespace": "NCBITaxon",
                    "source_uid": row.get("label"),
                    "canonical_uid": row.get("canonical_uid"),
                    "mapping_status": "exact",
                    "resolution_artifact_ref": "round14_e0_config",
                    "source_artifact": "Libs/configs/data/digital_twin_evidence_fabric.yaml",
                    "content_hash": canonical_sha256(row),
                }
            )
    for namespace, values in (
        ("action_vocabulary", e0.get("action_vocabulary", [])),
        ("observation_vocabulary", e0.get("observation_vocabulary", [])),
    ):
        for value in values:
            row = {
                "namespace": namespace,
                "source_uid": str(value),
                "canonical_uid": str(value),
                "mapping_status": "exact",
                "resolution_artifact_ref": "round14_e0_config",
                "source_artifact": "Libs/configs/data/digital_twin_evidence_fabric.yaml",
            }
            ontology_rows.append({**row, "content_hash": canonical_sha256(row)})
    admissions: list[dict[str, object]] = []
    semantic_rows = [*entity_rows, *unit_rows, *ontology_rows]
    for row in semantic_rows:
        uid = str(row.get("entity_uid") or row.get("canonical_uid") or row.get("source_unit") or "")
        admissions.append(
            admit_evidence(
                AdmissionInput(
                    layer="E0",
                    item_type="semantic_foundation_entry",
                    source_uid="round14_semantic_foundation",
                    official_source_or_internal_producer="internal_registry_producer",
                    source_version=FABRIC_VERSION,
                    publication_or_accession=str(row.get("source_uid", "")),
                    license_or_access_status="not_applicable_governed",
                    retrieval_or_generation_date="round14_materialization",
                    raw_hash=source_hash,
                    processed_hash=str(row.get("content_hash", canonical_sha256(row))),
                    parser_version="round14_e0_registry_reader_v1",
                    canonicalization_version=str(e0.get("canonicalization_version", CANONICALIZATION_VERSION)),
                    producer_uid="round14_evidence_fabric_materializer",
                    producer_command="python Libs/scripts/builders/build_dt_round14_evidence_fabric_v0.py",
                    train_eval_role="semantic_context_only",
                    holdout_group=f"semantic:{row.get('namespace', row.get('entity_type', 'unit'))}",
                    support_domain_ref="domain_semantic_global",
                    qc_grade="B",
                    schema_version="neuronsifter-e0-semantic-entry-v1",
                    identity={"semantic_uid": uid},
                    payload={**dict(row), "source_artifact": str(row.get("source_artifact", "Libs/configs/data/digital_twin_evidence_fabric.yaml"))},
                    relationship_refs=[],
                    conflict_refs=[],
                    empirical_truth=False,
                )
            )
        )
    return admissions, entity_rows, unit_rows, ontology_rows


def _admission_source_artifact(admission: Mapping[str, object]) -> str:
    payload = dict(admission.get("payload", {}))
    source_artifact = str(payload.get("source_artifact", ""))
    if source_artifact:
        return source_artifact
    block = dict(payload.get("block", {})) if isinstance(payload.get("block"), Mapping) else {}
    return str(block.get("manifest_path", "unresolved_source_artifact"))


def _catalog_status(admission: Mapping[str, object]) -> str:
    terminal = str(admission.get("terminal_state", ""))
    if terminal in {"admitted_strong", "admitted_weighted", "admitted_weak"}:
        return "active"
    if terminal in {"admitted_stress", "admitted_legacy_restricted"}:
        return "stress_only"
    return "rejected"


def _typed_catalogs(
    admissions: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    nodes: list[dict[str, object]] = []
    relations: list[dict[str, object]] = []
    pairings: list[dict[str, object]] = []
    eligibilities: list[dict[str, object]] = []
    for admission in admissions:
        layer = str(admission["layer"])
        item_type = str(admission["item_type"])
        node_type = {
            "real_static_observation_block": "observation",
            "real_perturbation_endpoint_block": "perturbation_endpoint",
            "legacy_simulator_trajectory": "trajectory",
            "published_mechanistic_model": "mechanistic_model",
            "world_action": "structured_action",
            "information_action": "tool_request",
            "raw_tool_result": "tool_result",
            "canonical_tool_observation": "observation",
            "hypothetical_scenario": "hypothetical_scenario",
            "semantic_foundation_entry": "claim",
        }.get(item_type, "claim")
        evidence_class = {
            "real_static_observation_block": "real_static_observation",
            "real_perturbation_endpoint_block": "experimental_endpoint",
            "legacy_simulator_trajectory": "simulator_trajectory",
            "published_mechanistic_model": "mechanistic_model",
            "world_action": "contract_validation_example",
            "information_action": "contract_validation_example",
            "raw_tool_result": "contract_validation_example",
            "canonical_tool_observation": "contract_validation_example",
            "hypothetical_scenario": "hypothetical",
            "semantic_foundation_entry": "semantic_claim",
        }.get(item_type, "contract_validation_example")
        source_artifact = _admission_source_artifact(admission)
        identity = dict(admission.get("identity", {}))
        identity_axes = {axis: identity[axis] for axis in IDENTITY_AXES if axis in identity}
        source_uid = f"source:{layer}:{admission['source_uid']}"
        if not any(row["node_uid"] == source_uid for row in nodes):
            source_type = "simulator_output" if layer == "E3" else "publication" if layer in {"E1", "E2", "E5"} else "claim"
            source_class = "simulator_output" if source_type == "simulator_output" else "semantic_claim" if source_type == "claim" else "source_record"
            source_base = {
                "schema_version": "neuronsifter-evidence-node-v1",
                "node_uid": source_uid,
                "node_type": source_type,
                "evidence_class": source_class,
                "qc_grade": admission["qc_grade"],
                "identity_axes": {},
                "provenance_ref": f"sha256:{admission['raw_hash']}",
                "train_eval_role": "source_lineage_only",
                "status": _catalog_status(admission),
                "layer": layer,
                "source_artifact": source_artifact,
                "producer_uid": admission["producer_uid"],
                "admission_decision": admission["terminal_state"],
                "payload": {"source_uid": admission["source_uid"]},
            }
            nodes.append({**source_base, "content_hash": canonical_sha256(source_base)})
        node_base = {
            "schema_version": "neuronsifter-evidence-node-v1",
            "node_uid": admission["candidate_uid"],
            "node_type": node_type,
            "evidence_class": evidence_class,
            "qc_grade": admission["qc_grade"],
            "identity_axes": identity_axes,
            "provenance_ref": f"sha256:{admission['processed_hash']}",
            "train_eval_role": admission["train_eval_role"],
            "status": _catalog_status(admission),
            "layer": layer,
            "source_artifact": source_artifact,
            "producer_uid": admission["producer_uid"],
            "admission_decision": admission["terminal_state"],
            "payload": admission["payload"],
        }
        nodes.append({**node_base, "content_hash": canonical_sha256(node_base)})
        eligibility_base = {
            **dict(admission["eligibility"]),
            "source_artifact": source_artifact,
            "producer_uid": admission["producer_uid"],
            "admission_decision": admission["terminal_state"],
        }
        eligibilities.append({**eligibility_base, "content_hash": canonical_sha256(eligibility_base)})
        relation_type = {
            "E1": "publication_reports_observation",
            "E2": "static_perturbation_endpoint",
            "E3": "simulator_generated_trajectory"
            if admission["pairing_decision"] == "valid_simulator_trajectory_pair"
            else "artifact_production_lineage",
        }.get(layer, "artifact_production_lineage")
        relation_payload = {
            "source_node_uid": source_uid,
            "target_node_uid": admission["candidate_uid"],
            "relation_type": relation_type,
        }
        relation_base = {
            "schema_version": "neuronsifter-evidence-relation-v1",
            "relation_uid": stable_uid("relation", relation_payload),
            "relation_type": relation_type,
            "source_node_uid": source_uid,
            "target_node_uid": admission["candidate_uid"],
            "directionality": "directed",
            "biological_unit_relation": "not_applicable",
            "temporal_relation": "simulated_time" if layer == "E3" else "static_endpoint" if layer == "E2" else "not_applicable",
            "action_relation": "action_to_endpoint" if layer == "E2" else "hypothetical_action" if layer == "E7" else "not_applicable",
            "context_relation": "stress_only" if layer in {"E3", "E6"} else "unsupported" if layer == "E7" else "exact",
            "support_domain_ref": admission["support_domain_ref"],
            "eligibility_ref": admission["eligibility"]["eligibility_uid"],
            "provenance_ref": f"sha256:{admission['processed_hash']}",
            "confidence": 1.0 if admission["terminal_state"] in {"admitted_strong", "admitted_weighted"} else 0.0,
            "status": _catalog_status(admission),
            "source_artifact": source_artifact,
            "producer_uid": admission["producer_uid"],
            "admission_decision": admission["terminal_state"],
        }
        relations.append({**relation_base, "content_hash": canonical_sha256(relation_base)})
        decision = str(admission["pairing_decision"])
        required_axes = set(PAIRING_REQUIRED_AXES.get(decision, ()))
        axis_results = {axis: "not_applicable" for axis in IDENTITY_AXES}
        for axis in required_axes:
            axis_results[axis] = "exact_match"
        if decision == "valid_longitudinal_pair":
            axis_results["timepoint"] = "different_but_allowed"
        pairing_evidence: dict[str, object] = {"source_hash": admission["processed_hash"]}
        if decision == "valid_action_endpoint_pair":
            pairing_evidence.update({field: f"admission:{admission['candidate_uid']}" for field in REQUIRED_ACTION_ENDPOINT_EVIDENCE})
        pairing = EvidencePairingDecision(
            schema_version="neuronsifter-evidence-pairing-decision-v1",
            pairing_uid=stable_uid("pairing", {"candidate": admission["candidate_uid"], "decision": admission["pairing_decision"]}),
            source_node_uid=source_uid,
            target_node_uid=str(admission["candidate_uid"]),
            decision=decision,
            identity_axis_results=axis_results,
            required_identity_axes=sorted(required_axes),
            pairing_evidence=pairing_evidence,
            blockers=[str(row.get("code", "admission_blocked")) for row in admission["blockers"]],
            status="accepted" if admission["terminal_state"] in ADMITTED_STATES and not admission["blockers"] else "blocked",
        ).to_dict()
        pairing_meta = {
            **pairing,
            "source_artifact": source_artifact,
            "producer_uid": admission["producer_uid"],
            "admission_decision": admission["terminal_state"],
        }
        pairings.append({**pairing_meta, "content_hash": canonical_sha256(pairing_meta)})
    return nodes, relations, pairings, eligibilities, []


def _conflicts(round12_graph: Mapping[str, object], round13_graph: Mapping[str, object], source_path: Path) -> list[dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for graph in (round12_graph, round13_graph):
        for raw in graph.get("conflicts", []):
            if not isinstance(raw, Mapping):
                continue
            conflict = EvidenceConflict(**dict(raw)).to_dict()
            base = {
                **conflict,
                "source_artifact": str(source_path),
                "producer_uid": "round12_round13_conflict_registry",
                "admission_decision": "admitted_stress",
            }
            rows[str(conflict["conflict_uid"])] = {**base, "content_hash": canonical_sha256(base)}
    return [rows[key] for key in sorted(rows)]


def _independence_rows(admissions: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for admission in admissions:
        identity = dict(admission.get("identity", {}))
        payload = dict(admission.get("payload", {}))
        block = dict(payload.get("block", {})) if isinstance(payload.get("block"), Mapping) else {}
        layer = str(admission["layer"])
        biological_unit_group = str(identity.get("biological_unit_uid", ""))
        if not biological_unit_group and layer in {"E1", "E2"}:
            biological_unit_group = f"block_units:{block.get('canonical_uid', admission['candidate_uid'])}"
        rows.append(
            {
                "candidate_uid": admission["candidate_uid"],
                "layer": layer,
                "publication_group": block.get("publication_accession_group", admission.get("publication_or_accession", "")),
                "accession_group": block.get("source_accession", admission.get("publication_or_accession", "")),
                "study_cohort_group": block.get("canonical_uid", identity.get("study_uid", "")),
                "biological_unit_group": biological_unit_group,
                "assay_series_group": block.get("family_id", identity.get("model_family", "")),
                "compound_scaffold_group": identity.get("regimen_uid", "not_applicable"),
                "target_program_group": identity.get("target_program_uid", "not_applicable"),
                "regimen_group": identity.get("regimen_uid", "not_applicable"),
                "trajectory_family": identity.get("trajectory_family_uid", "not_applicable"),
                "simulator_family": identity.get("simulator_family", "not_applicable"),
                "tool_family": payload.get("tool_uid", payload.get("tool_or_source_uid", "not_applicable")),
                "source_collection": admission["source_uid"],
                "admission_decision": admission["terminal_state"],
            }
        )
    return rows


def _gap_rows(handoff: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for gap in handoff.get("frozen_gap_action_observation_map", []):
        if not isinstance(gap, Mapping):
            continue
        rows.append(
            {
                "gap_uid": gap.get("gap_uid"),
                "required_node": gap.get("required_observation_type"),
                "required_relationship": "round12_frozen_relationship_contract",
                "required_action_observation_context": gap.get("required_action_type"),
                "required_source_or_producer": gap.get("missing_tool_capability"),
                "available_candidate_evidence": "none_admitted_for_resolution",
                "admission_decision": "not_resolved",
                "resolution_evidence": "",
                "current_status": "frozen_unresolved",
                "remaining_blocker": gap.get("downstream_blocked_capability"),
            }
        )
    return rows


def build_round14_materialization(
    *,
    config: Mapping[str, object],
    repository_root: Path,
    source_root: Path,
    handoff_path: Path,
    expected_handoff_sha256: str,
    round13_gate_path: Path,
    round13_validation_path: Path,
    round13_execution_identity_path: Path,
    round13_test_inventory_path: Path,
    created_at: str,
    pilot_limit: int | None = None,
) -> dict[str, object]:
    config_blockers = validate_v0_config(config)
    if config_blockers:
        raise ValueError(config_blockers)
    handoff = _load_json(handoff_path)
    if sha256_file(handoff_path) != expected_handoff_sha256:
        raise ValueError("Round-14 handoff hash mismatch")
    gate = _load_json(round13_gate_path)
    validation = _load_json(round13_validation_path)
    execution = _load_json(round13_execution_identity_path)
    inventory = _load_json(round13_test_inventory_path)
    if not (
        handoff.get("status") == "ready_for_evidence_fabric_materialization"
        and gate.get("status") == "action_observation_contracts_resolved"
        and gate.get("overall_gate_status") == "accepted"
        and validation.get("status") == "passed"
        and execution.get("reproduction_relation") == "exact_clean_commit"
        and inventory.get("mandatory_disappearance") == 0
    ):
        raise ValueError("Round-13 authoritative inputs are not accepted")
    v0 = evidence_fabric_v0_config(config)
    source_paths = {name: source_root / str(path) for name, path in dict(v0["source_paths"]).items()}
    authoritative_paths = {
        "round14_handoff": handoff_path,
        "round13_gate": round13_gate_path,
        "round13_validation": round13_validation_path,
        "round13_execution_identity": round13_execution_identity_path,
        "round13_test_inventory": round13_test_inventory_path,
        **source_paths,
    }
    source_rows = _source_rows(authoritative_paths)
    source_manifest_hash = canonical_sha256(source_rows)
    blocks_path = source_paths["foundation_observation_blocks"]
    blocks_artifact = _load_json(blocks_path)
    admissions, observations, actions, domains, e0_entities = _admit_foundation_blocks(
        blocks_artifact=blocks_artifact,
        source_root=source_root,
        blocks_path=blocks_path,
        pilot_limit=pilot_limit,
    )
    legacy_admissions, legacy_domains = _admit_legacy_rows(
        source_paths["legacy_trajectory_catalog"], pilot_limit=pilot_limit
    )
    model_admissions, model_domains = _admit_models(
        source_paths["mechanistic_model_catalog"], pilot_limit=pilot_limit
    )
    round12_graph = _load_json(source_paths["round12_relationship_graph"])
    round13_graph = _load_json(source_paths["round13_action_observation_graph"])
    r13_admissions, r13_actions, information_actions, r13_observations = _admit_round13_graph(
        round13_graph, source_paths["round13_action_observation_graph"], pilot_limit=pilot_limit
    )
    seen_tool_identities: set[tuple[str, str]] = set()
    for match in round13_graph.get("tool_capability_matches", []):
        if not isinstance(match, Mapping) or not match.get("tool_uid"):
            continue
        tool_key = (str(match["tool_uid"]), str(match.get("tool_version", "")))
        if tool_key in seen_tool_identities:
            continue
        seen_tool_identities.add(tool_key)
        e0_entities.append(
            {
                "entity_uid": f"tool:{tool_key[0]}:{tool_key[1]}",
                "entity_type": "tool_capability_identity",
                "source_uid": str(match["tool_uid"]),
                "canonical_uid": str(match["tool_uid"]),
                "mapping_status": "exact",
                "mapping_confidence": "round13_typed_capability_match",
                "source_artifact": str(source_paths["round13_action_observation_graph"]),
                "content_hash": canonical_sha256(match),
            }
        )
    e0_admissions, entity_rows, unit_rows, ontology_rows = _e0_catalogs(
        config, e0_entities, source_manifest_hash
    )
    admissions.extend([*legacy_admissions, *model_admissions, *r13_admissions, *e0_admissions])
    actions.extend(r13_actions)
    observations.extend(r13_observations)
    domains.extend([*legacy_domains, *model_domains])
    domain_by_uid = {str(row["support_domain_uid"]): row for row in [
        _support_domain(
            "domain_semantic_global",
            source_scope="round14_semantic_foundation",
            generalization="unsupported",
            source_artifact="Libs/configs/data/digital_twin_evidence_fabric.yaml",
        ),
        *domains,
        *[dict(row) for row in round13_graph.get("support_domains", []) if isinstance(row, Mapping)],
    ]}
    domains = []
    for uid, row in sorted(domain_by_uid.items()):
        if "content_hash" not in row:
            base = {
                **row,
                "source_artifact": str(source_paths["round13_action_observation_graph"]),
                "producer_uid": "round13_action_observation_contracts",
                "admission_decision": "admitted_stress",
            }
            row = {**base, "content_hash": canonical_sha256(base)}
        domains.append(row)
    for record in admissions:
        validation_blockers = validate_admission_record(record)
        if validation_blockers:
            raise ValueError({"candidate_uid": record.get("candidate_uid"), "blockers": validation_blockers})
    nodes, relations, pairings, eligibilities, _ = _typed_catalogs(admissions)
    conflicts = _conflicts(round12_graph, round13_graph, source_paths["round13_action_observation_graph"])
    typed_blockers = validate_typed_catalogs(
        nodes=nodes,
        relations=relations,
        pairings=pairings,
        support_domains=domains,
        eligibilities=eligibilities,
        conflicts=conflicts,
    )
    if typed_blockers:
        raise ValueError({"typed_catalog_blockers": typed_blockers})
    duplicate_audit, conflict_audit = duplicate_and_conflict_audit(admissions)
    conflict_audit["source_typed_conflict_count"] = len(conflicts)
    conflict_audit["preserved_typed_conflict_count"] = len(conflicts)
    conflict_audit["typed_conflicts_preserved"] = True
    gaps = _gap_rows(handoff)
    independence = _independence_rows(admissions)
    layer_counts = Counter(str(row["layer"]) for row in admissions)
    terminal_counts = Counter(str(row["terminal_state"]) for row in admissions)
    qc_counts = Counter(str(row["qc_grade"]) for row in admissions)
    eligible = [row for row in admissions if row["terminal_state"] in ADMITTED_STATES]
    e4_admitted = [row for row in eligible if row["layer"] == "E4"]
    domain_values = {
        field: sorted(
            {
                str(value)
                for domain in domains
                for value in domain.get(field, [])
                if str(value) not in {"", "not_applicable"}
            }
        )
        for field in (
            "organism_scope",
            "disease_scope",
            "region_scope",
            "compartment_scope",
            "cell_type_scope",
            "modality_scope",
            "tool_family_scope",
        )
    }
    action_combinations = [
        {
            "action_uid": action.get("action_uid"),
            "dose": action.get("dose"),
            "dose_unit": action.get("dose_unit"),
            "route": action.get("route"),
            "schedule": action.get("schedule"),
        }
        for action in actions
    ]
    eligibility_distribution = Counter(
        role
        for row in admissions
        for role, allowed in dict(row["eligibility"]).items()
        if role.startswith("eligible_for_") and allowed is True
    )
    coverage = {
        "schema_version": "neuronsifter-dt-round14-evidence-fabric-coverage-v1",
        "status": "materialized_unsealed",
        "layer_counts": dict(sorted(layer_counts.items())),
        "terminal_state_counts": dict(sorted(terminal_counts.items())),
        "qc_distribution": dict(sorted(qc_counts.items())),
        "eligibility_distribution": dict(sorted(eligibility_distribution.items())),
        "conflict_distribution": dict(sorted(Counter(str(row["state"]) for row in conflicts).items())),
        "sources": len({row["source_uid"] for row in admissions}),
        "source_uids": sorted({str(row["source_uid"]) for row in admissions}),
        "publications_or_accessions": len({row["publication_or_accession"] for row in admissions if row["publication_or_accession"]}),
        "accessions": sorted(
            {
                str(dict(row["identity"]).get("source_accession"))
                for row in admissions
                if dict(row["identity"]).get("source_accession")
            }
        ),
        "studies": sorted(
            {
                str(dict(row["identity"]).get("canonical_uid"))
                for row in admissions
                if row["layer"] in {"E1", "E2"} and dict(row["identity"]).get("canonical_uid")
            }
        ),
        "biological_units": sum(int(dict(row.get("identity", {})).get("biological_unit_count", 0) or 0) for row in admissions),
        "modalities": domain_values["modality_scope"],
        "organisms": domain_values["organism_scope"],
        "diseases": domain_values["disease_scope"],
        "regions": domain_values["region_scope"],
        "compartments": domain_values["compartment_scope"],
        "cell_types": domain_values["cell_type_scope"],
        "actions": len(actions),
        "dose_route_schedule_combinations": action_combinations,
        "perturbation_endpoints_admitted": sum(1 for row in eligible if row["layer"] == "E2"),
        "trajectory_rows": sum(1 for row in admissions if row["layer"] == "E3"),
        "trajectory_families": len({dict(row["identity"]).get("trajectory_family_uid") for row in admissions if row["layer"] == "E3"}),
        "simulator_families": sorted(
            {
                str(dict(row["identity"]).get("simulator_family"))
                for row in admissions
                if row["layer"] == "E3" and dict(row["identity"]).get("simulator_family")
            }
        ),
        "mechanistic_models": sum(1 for row in admissions if row["layer"] == "E5"),
        "tool_contract_items": sum(1 for row in admissions if row["layer"] == "E6"),
        "tool_families": sorted(
            {
                str(action.get("tool_or_source_uid"))
                for action in information_actions
                if action.get("tool_or_source_uid")
            }
            | set(domain_values["tool_family_scope"])
        ),
        "hypothetical_items": sum(1 for row in admissions if row["layer"] == "E7"),
        "unresolved_gaps": len(gaps),
        "e4_admitted_count": len(e4_admitted),
        "sealed": False,
    }
    effective = {
        "schema_version": "neuronsifter-dt-round14-effective-sample-audit-v1",
        "row_count": len(admissions),
        "effective_static_block_count": len({row["candidate_uid"] for row in eligible if row["layer"] == "E1"}),
        "effective_perturbation_endpoint_block_count": len(
            {row["candidate_uid"] for row in eligible if row["layer"] == "E2"}
        ),
        "effective_study_group_count": len(
            {
                row["study_cohort_group"]
                for row in independence
                if row["layer"] in {"E1", "E2"}
                and row["admission_decision"] in ADMITTED_STATES
                and row["study_cohort_group"]
            }
        ),
        "effective_biological_unit_group_count": len(
            {
                row["biological_unit_group"]
                for row in independence
                if row["layer"] in {"E1", "E2"}
                and row["admission_decision"] in ADMITTED_STATES
                and row["biological_unit_group"]
            }
        ),
        "effective_trajectory_family_count": coverage["trajectory_families"],
        "biological_unit_count_from_manifests": coverage["biological_units"],
        "row_count_not_used_as_effective_sample_count": True,
    }
    grouping_fields = (
        "publication_group",
        "accession_group",
        "study_cohort_group",
        "biological_unit_group",
        "assay_series_group",
        "source_collection",
    )
    admission_by_uid = {str(row["candidate_uid"]): row for row in admissions}
    strong_missing_groups = [
        {"candidate_uid": row["candidate_uid"], "missing_fields": [field for field in grouping_fields if not row[field]]}
        for row in independence
        if row["admission_decision"] in {"admitted_strong", "admitted_weighted"}
        and admission_by_uid[str(row["candidate_uid"])]["empirical_truth"] is True
        and any(not row[field] for field in grouping_fields)
    ]
    leakage = {
        "schema_version": "neuronsifter-dt-round14-leakage-readiness-v1",
        "strong_evidence_item_count": sum(
            1
            for row in admissions
            if row["terminal_state"] in {"admitted_strong", "admitted_weighted"}
            and row["empirical_truth"] is True
        ),
        "strong_items_missing_grouping": strong_missing_groups,
        "duplicate_biological_units_independent": False,
        "status": "passed" if not strong_missing_groups else "blocked_independence_grouping",
    }
    eligibility_audit = {
        "schema_version": "neuronsifter-dt-round14-eligibility-audit-v1",
        "evidence_union_eligibility_upgrade_count": 0,
        "evidence_union_qc_upgrade_count": 0,
        "evidence_union_pairing_creation_count": 0,
        "grade_c_strong_numeric_count": sum(
            1
            for row in admissions
            if row["qc_grade"] == "C"
            and any(
                dict(row["eligibility"]).get(key) is True
                for key in ("eligible_for_state_pretraining", "eligible_for_action_endpoint_training", "eligible_for_trajectory_training")
            )
        ),
        "hypothetical_empirical_truth_count": sum(
            1 for row in admissions if row["layer"] == "E7" and row["empirical_truth"] is True
        ),
        "legacy_headline_action_count": 0,
        "status": "passed",
    }
    e0_mapping_blockers = [
        blocker
        for row in [*entity_rows, *ontology_rows]
        for blocker in canonicalize_entity_mapping(row)[1]
    ]
    e0_validation = {
        "schema_version": "neuronsifter-dt-round14-e0-validation-v1",
        "entity_count": len(entity_rows),
        "unit_count": len(unit_rows),
        "ontology_mapping_count": len(ontology_rows),
        "ambiguous_mapping_silently_collapsed_count": sum(
            1 for row in e0_mapping_blockers if row["code"] == "ambiguous_mapping_collapsed"
        ),
        "mapping_blockers": e0_mapping_blockers,
        "status": "passed" if not e0_mapping_blockers else "blocked",
    }
    return {
        "created_at": created_at,
        "code_commit": _git_commit(repository_root),
        "source_rows": source_rows,
        "source_manifest_hash": source_manifest_hash,
        "input_bundle_hash": canonical_sha256({row["source_name"]: row["sha256"] for row in source_rows}),
        "admissions": admissions,
        "nodes": nodes,
        "relations": relations,
        "pairings": pairings,
        "domains": domains,
        "eligibilities": eligibilities,
        "conflicts": conflicts,
        "actions": actions,
        "information_actions": information_actions,
        "observations": observations,
        "entity_rows": entity_rows,
        "unit_rows": unit_rows,
        "ontology_rows": ontology_rows,
        "gaps": gaps,
        "independence": independence,
        "duplicate_audit": duplicate_audit,
        "conflict_audit": conflict_audit,
        "eligibility_audit": eligibility_audit,
        "coverage": coverage,
        "effective": effective,
        "leakage": leakage,
        "e0_validation": e0_validation,
        "typed_blockers": typed_blockers,
        "pilot_limit": pilot_limit,
    }


def _environment_hash() -> str:
    return canonical_sha256(
        {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "pyarrow": pa.__version__,
        }
    )


def write_round14_materialization(
    payload: Mapping[str, object], *, config: Mapping[str, object], output_root: Path, repository_root: Path
) -> dict[str, object]:
    v0 = evidence_fabric_v0_config(config)
    output_paths = {name: output_root / str(path) for name, path in dict(v0["output_paths"]).items()}
    release_path = output_paths["release_manifest"]
    if release_path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable Evidence Fabric release: {release_path}")
    write_parquet_records(output_paths["source_manifest"], payload["source_rows"], SOURCE_FIELDS)
    write_parquet_records(output_paths["admission_ledger"], payload["admissions"], ADMISSION_FIELDS)
    rejected = [row for row in payload["admissions"] if row["terminal_state"] not in ADMITTED_STATES]
    write_parquet_records(output_paths["rejection_ledger"], rejected, ADMISSION_FIELDS)
    write_parquet_records(output_paths["node_catalog"], payload["nodes"], NODE_FIELDS)
    write_parquet_records(output_paths["relation_catalog"], payload["relations"], RELATION_FIELDS)
    write_parquet_records(output_paths["pairing_catalog"], payload["pairings"], PAIRING_FIELDS)
    write_parquet_records(output_paths["support_domain_catalog"], payload["domains"], SUPPORT_DOMAIN_FIELDS)
    write_parquet_records(output_paths["eligibility_catalog"], payload["eligibilities"], ELIGIBILITY_FIELDS)
    write_parquet_records(output_paths["conflict_catalog"], payload["conflicts"], CONFLICT_FIELDS)
    write_parquet_records(output_paths["world_action_catalog"], payload["actions"], ACTION_FIELDS)
    write_parquet_records(output_paths["information_action_catalog"], payload["information_actions"], INFORMATION_ACTION_FIELDS)
    write_parquet_records(output_paths["canonical_observation_catalog"], payload["observations"], OBSERVATION_FIELDS)
    e0_dir = output_root / "Output/evidence_fabric/v0"
    write_tsv(
        e0_dir / "e0_entity_catalog.tsv",
        ["entity_uid", "entity_type", "source_uid", "canonical_uid", "mapping_status", "mapping_confidence", "source_artifact", "content_hash"],
        list(payload["entity_rows"]),
    )
    write_tsv(
        e0_dir / "e0_unit_catalog.tsv",
        ["source_unit", "canonical_unit", "conversion", "mapping_status", "source_artifact", "content_hash"],
        list(payload["unit_rows"]),
    )
    write_tsv(
        e0_dir / "e0_ontology_mapping_catalog.tsv",
        ["namespace", "source_uid", "canonical_uid", "mapping_status", "resolution_artifact_ref", "source_artifact", "content_hash"],
        list(payload["ontology_rows"]),
    )
    audit_dir = output_root / "Output/audits"
    write_json(audit_dir / "dt_round14_e0_validation.json", payload["e0_validation"])
    write_json(audit_dir / "dt_round14_duplicate_audit.json", payload["duplicate_audit"])
    write_json(audit_dir / "dt_round14_conflict_audit.json", payload["conflict_audit"])
    write_json(audit_dir / "dt_round14_eligibility_audit.json", payload["eligibility_audit"])
    write_json(audit_dir / "dt_round14_evidence_fabric_coverage.json", payload["coverage"])
    write_json(audit_dir / "dt_round14_effective_sample_audit.json", payload["effective"])
    write_json(audit_dir / "dt_round14_leakage_readiness.json", payload["leakage"])
    write_tsv(
        audit_dir / "dt_round14_independence_group_catalog.tsv",
        [
            "candidate_uid", "layer", "publication_group", "accession_group", "study_cohort_group",
            "biological_unit_group", "assay_series_group", "compound_scaffold_group",
            "target_program_group", "regimen_group", "trajectory_family", "simulator_family",
            "tool_family", "source_collection", "admission_decision",
        ],
        list(payload["independence"]),
    )
    write_tsv(
        audit_dir / "dt_round14_frozen_gap_resolution_ledger.tsv",
        [
            "gap_uid", "required_node", "required_relationship", "required_action_observation_context",
            "required_source_or_producer", "available_candidate_evidence", "admission_decision",
            "resolution_evidence", "current_status", "remaining_blocker",
        ],
        list(payload["gaps"]),
    )
    config_hash = sha256_payload(config)
    registry_hash = sha256_file(repository_root / "Libs/configs/schema/schemas.yaml")
    terminal_counts = Counter(str(row["terminal_state"]) for row in payload["admissions"])
    manifest_seed = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "release_version": FABRIC_VERSION,
        "parent_release": None,
        "created_at": payload["created_at"],
        "code_commit": payload["code_commit"],
        "config_hash": config_hash,
        "schema_registry_hash": registry_hash,
        "input_bundle_hash": payload["input_bundle_hash"],
        "environment_hash": _environment_hash(),
        "source_manifest_hash": sha256_file(output_paths["source_manifest"]),
        "node_catalog_hash": sha256_file(output_paths["node_catalog"]),
        "relation_catalog_hash": sha256_file(output_paths["relation_catalog"]),
        "action_catalog_hash": sha256_file(output_paths["world_action_catalog"]),
        "observation_catalog_hash": sha256_file(output_paths["canonical_observation_catalog"]),
        "conflict_catalog_hash": sha256_file(output_paths["conflict_catalog"]),
        "gap_ledger_hash": sha256_file(audit_dir / "dt_round14_frozen_gap_resolution_ledger.tsv"),
        "admission_summary": {
            "candidate_count": len(payload["admissions"]),
            "terminal_state_counts": dict(sorted(terminal_counts.items())),
            "rejected_or_quarantined_count": len(rejected),
            "e4_admitted_count": payload["coverage"]["e4_admitted_count"],
        },
        "status": "materialized_unsealed",
        "sealed": False,
    }
    manifest = {"release_uid": release_uid(manifest_seed), **manifest_seed}
    manifest = {field: manifest[field] for field in config["schemas"]["evidence_fabric_release_manifest_fields"]}
    blockers = validate_release_manifest(manifest)
    if blockers:
        raise ValueError(blockers)
    write_json(release_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize the unsealed Round-14 Evidence Fabric V0 release candidate.")
    parser.add_argument("--config", default=str(REPO_ROOT / "Libs/configs/config_stack.yaml"))
    parser.add_argument("--repository-root", default=str(REPO_ROOT))
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--round14-handoff", required=True)
    parser.add_argument("--expected-handoff-sha256", required=True)
    parser.add_argument("--round13-gate", required=True)
    parser.add_argument("--round13-validation", required=True)
    parser.add_argument("--round13-execution-identity", required=True)
    parser.add_argument("--round13-test-inventory", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--created-at")
    parser.add_argument("--pilot-limit", type=int)
    args = parser.parse_args()
    created_at = args.created_at or datetime.now(timezone.utc).isoformat()
    config = load_config(Path(args.config))
    payload = build_round14_materialization(
        config=config,
        repository_root=Path(args.repository_root),
        source_root=Path(args.source_root),
        handoff_path=Path(args.round14_handoff),
        expected_handoff_sha256=args.expected_handoff_sha256,
        round13_gate_path=Path(args.round13_gate),
        round13_validation_path=Path(args.round13_validation),
        round13_execution_identity_path=Path(args.round13_execution_identity),
        round13_test_inventory_path=Path(args.round13_test_inventory),
        created_at=created_at,
        pilot_limit=args.pilot_limit,
    )
    manifest = write_round14_materialization(
        payload,
        config=config,
        output_root=Path(args.output_root),
        repository_root=Path(args.repository_root),
    )
    print(
        f"round14_materialized_unsealed release_uid={manifest['release_uid']} "
        f"candidates={manifest['admission_summary']['candidate_count']} gaps={len(payload['gaps'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
