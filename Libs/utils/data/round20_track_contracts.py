"""Fail-closed validation for the governed Round-20 Track A/B/C contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from Libs.utils.data.round19_contracts import LABEL_CLASSES, READINESS_STATES
from Libs.utils.data.trajectory_contracts import blocker, is_sha256, sorted_blockers


TRACK_NAMES = frozenset(
    {
        "track_a_allen_full_landing",
        "track_b_public_e2_e4_recovery",
        "track_c_runtime_deployment",
    }
)
ALLEN_FAMILY_UIDS = frozenset(
    {
        "allen_abc_mouse",
        "allen_abc_human",
        "allen_abc_human_ad",
        "allen_cell_types_transcriptomics",
        "allen_cell_types_morphology",
        "allen_cell_types_ephys",
        "allen_human_brain_microarray",
        "allen_human_brain_mri",
        "allen_mouse_connectivity",
        "allen_developing_mouse",
        "allen_brain_observatory",
        "allen_reference_atlas_ccf",
    }
)
RIGHTS_FIELDS = frozenset(
    {
        "family_uid",
        "dataset_and_release",
        "official_host",
        "payload_host",
        "governing_terms",
        "dataset_specific_license",
        "third_party_license",
        "citation_policy",
        "internal_download_allowed",
        "internal_transformation_allowed",
        "internal_training_allowed",
        "aggregate_reporting_allowed",
        "raw_redistribution_allowed",
        "processed_redistribution_allowed",
        "derived_feature_release_allowed",
        "checkpoint_release_allowed",
        "commercial_use_status",
        "required_attribution",
        "decision",
        "blocker",
    }
)
RIGHTS_DECISIONS = frozenset(
    {
        "approved_internal_noncommercial",
        "approved_with_release_restrictions",
        "approved_internal_and_derived_release",
        "blocked_third_party_terms",
        "blocked_missing_release_identity",
        "blocked_terms_ambiguity",
    }
)
RELEASE_FIELDS = frozenset(
    {
        "release_uid",
        "release_date_or_version",
        "official_landing_page",
        "api_or_download_endpoint",
        "object_inventory",
        "format",
        "expected_size",
        "official_checksum",
        "expected_biological_object_counts",
        "pagination_or_request_identity",
        "etag_or_last_modified",
    }
)
ADAPTER_METHODS = frozenset(
    {"discover", "estimate", "retrieve", "resume", "verify", "register", "emit_source_card"}
)
PARSER_FAMILIES = frozenset(
    {
        "single_cell_transcriptomics",
        "spatial_transcriptomics",
        "microarray",
        "mri_anatomy_metadata",
        "nwb_electrophysiology",
        "swc_morphology",
        "connectivity_structure_tables",
        "connectivity_voxel_grids",
        "developmental_expression",
        "reference_atlas_volumes_and_ontologies",
        "brain_observatory_activity_and_stimulus_metadata",
    }
)
DETERMINISTIC_CONSUMERS = frozenset(
    {
        "region_hierarchy_builder",
        "ccf_graph_builder",
        "cell_taxonomy_graph_builder",
        "transcriptomics_morphology_ephys_linker",
        "human_mouse_alignment_builder",
        "donor_specimen_source_split_builder",
        "missing_modality_task_builder",
        "development_ood_task_builder",
    }
)
E2_FIELDS = frozenset(
    {
        "action",
        "comparator",
        "dose",
        "dose_unit",
        "route",
        "schedule",
        "timepoint",
        "context",
        "endpoint",
        "endpoint_unit",
        "sample_size",
        "uncertainty",
        "publication_or_accession",
        "raw_or_supplementary_source_hash",
    }
)
SERVICE_ENDPOINTS = frozenset(
    {
        "/infer_world_state",
        "/rollout_counterfactuals",
        "/plan_interventions",
        "/explain_decision",
        "/ingest_feedback",
        "/plan_tool_acquisition_batch",
        "/dispatch_tool_tasks",
        "/ingest_tool_result",
        "/active_learning_round_report",
    }
)
REGISTRY_NAMES = frozenset(
    {
        "source_registry",
        "license_registry",
        "entity_unit_crosswalk",
        "artifact_cas",
        "split_registry",
        "run_registry",
        "checkpoint_registry",
        "tool_registry",
        "report_builder",
    }
)
COMMON_TRACK_FIELDS = frozenset(
    {
        "owner",
        "status",
        "scope",
        "resource_budget",
        "test_membership",
        "pass_conditions",
        "block_conditions",
        "artifact_names",
        "authorization",
        "handoff",
    }
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: object) -> list[Any]:
    return list(value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else []


def _repository_path(
    repository: Path,
    value: object,
    *,
    label: str,
    blockers: list[dict[str, object]],
) -> Path | None:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        blockers.append(blocker("contract_path", "Contract paths must be non-empty repository-relative paths.", label=label))
        return None
    repository = repository.resolve()
    path = (repository / value).resolve()
    try:
        path.relative_to(repository)
    except ValueError:
        blockers.append(blocker("contract_path_escape", "Contract path escapes the repository.", label=label, path=value))
        return None
    return path


def _validate_hashed_inputs(
    rows: object,
    *,
    repository: Path,
    label: str,
    blockers: list[dict[str, object]],
) -> None:
    for index, item in enumerate(_sequence(rows)):
        row = _mapping(item)
        path = _repository_path(repository, row.get("path"), label=f"{label}[{index}]", blockers=blockers)
        expected_hash = row.get("sha256")
        if not is_sha256(expected_hash):
            blockers.append(blocker("input_hash", "Frozen input hash must be SHA-256.", label=label, index=index))
            continue
        if path is None:
            continue
        if not path.is_file():
            blockers.append(blocker("input_missing", "Frozen contract input is missing.", label=label, path=row.get("path")))
            continue
        actual_hash = file_sha256(path)
        if actual_hash != expected_hash:
            blockers.append(
                blocker(
                    "input_hash_drift",
                    "Frozen contract input hash drifted.",
                    label=label,
                    path=row.get("path"),
                    expected=expected_hash,
                    actual=actual_hash,
                )
            )


def historical_allen_summary(contract: Mapping[str, object], repository: Path) -> dict[str, object]:
    track = _mapping(_mapping(contract.get("tracks")).get("track_a_allen_full_landing"))
    inputs = {
        _mapping(item).get("kind"): repository / str(_mapping(item).get("path"))
        for item in _sequence(track.get("historical_inputs"))
    }
    cards_payload = json.loads(inputs["historical_source_cards"].read_text(encoding="utf-8"))
    cards = _sequence(cards_payload.get("records"))
    gate = json.loads(inputs["historical_failed_gate"].read_text(encoding="utf-8"))
    junit_root = ET.parse(inputs["historical_failed_test_execution"]).getroot()
    suites = list(junit_root.iter("testsuite"))
    junit = {
        key: sum(int(suite.attrib.get(key, "0")) for suite in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    return {
        "source_card_count": len(cards),
        "source_card_family_uids": sorted(str(_mapping(card).get("source_uid")) for card in cards),
        "source_cards_with_raw_hashes": sum(bool(_mapping(card).get("raw_hashes")) for card in cards),
        "source_cards_with_release_identity": sum(bool(_mapping(card).get("source_release_uid")) for card in cards),
        "historical_gate_status": gate.get("status"),
        "historical_junit": junit,
    }


def validate_round20_track_contracts(
    contract: Mapping[str, object], *, repository: Path
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    if contract.get("version") != "neuronsifter-round20-track-contracts-v1":
        blockers.append(blocker("contract_identity", "Round-20 track contract version is not frozen."))
    if contract.get("status") != "contract_frozen_before_track_execution" or contract.get("created_before_track_execution") is not True:
        blockers.append(blocker("contract_timing", "Track contracts must be frozen before execution."))
    if contract.get("base_main_commit") != "ee45298421c9dbabab952f790b4662e8c7d67f87":
        blockers.append(blocker("base_commit", "Track contracts do not bind to the accepted Round-20 main commit."))

    _validate_hashed_inputs(contract.get("upstream_artifacts"), repository=repository, label="upstream_artifacts", blockers=blockers)
    upstream = _mapping(contract.get("upstream_gate"))
    upstream_path = _repository_path(repository, upstream.get("path"), label="upstream_gate", blockers=blockers)
    if upstream_path and upstream_path.is_file():
        gate = json.loads(upstream_path.read_text(encoding="utf-8"))
        if gate.get("status") != "passed" or gate.get("gate_uid") != "round20_shared_schema_gate":
            blockers.append(blocker("upstream_gate", "The shared-schema Gate is not accepted."))
        if gate.get("gate_closure", {}).get("mainline_promotion_status") != "passed_isolated_ff_only":
            blockers.append(blocker("upstream_promotion", "The shared-schema Gate is not mainline-promoted."))
        if gate.get("allen_payload_download_authorized") is not False or gate.get("e2_e4_training_authorized") is not False:
            blockers.append(blocker("upstream_authorization", "Upstream Gate improperly authorizes payload download or training."))

    invariants = _mapping(contract.get("invariants"))
    expected_invariants = {
        "strict_public_only_flag": True,
        "controlled_source_resolution_count": 0,
        "final_held_out_access_count": 0,
        "training_authorized": False,
        "training_started": False,
        "evidence_fabric_sealed": False,
        "frozen_gap_count": 20,
        "dt_core_status": "not_evaluated",
        "remote_push_authorized": False,
    }
    if any(invariants.get(key) != value for key, value in expected_invariants.items()):
        blockers.append(blocker("invariant_drift", "A public, training, held-out, Evidence Fabric, DT_CORE, or push invariant drifted."))

    shared = _mapping(contract.get("shared_fact_sources"))
    if shared.get("owner") != "round20-shared-governance" or shared.get("independent_track_mutation_allowed") is not False:
        blockers.append(blocker("shared_owner", "Shared fact sources must retain one owner and prohibit independent track mutation."))

    tracks = _mapping(contract.get("tracks"))
    if set(tracks) != TRACK_NAMES:
        blockers.append(blocker("track_inventory", "Track A/B/C inventory is not exact.", actual=sorted(tracks)))
        return sorted_blockers(blockers)
    for name, value in tracks.items():
        track = _mapping(value)
        missing = sorted(COMMON_TRACK_FIELDS - set(track))
        if missing:
            blockers.append(blocker("track_fields", "Track contract lacks common fields.", track=name, missing=missing))
        if track.get("status") != "contract_frozen_execution_not_started":
            blockers.append(blocker("track_status", "Track execution status is not fail-closed.", track=name))

    track_a = _mapping(tracks["track_a_allen_full_landing"])
    _validate_hashed_inputs(track_a.get("historical_inputs"), repository=repository, label="track_a.historical_inputs", blockers=blockers)
    _validate_hashed_inputs([track_a.get("source_registry")], repository=repository, label="track_a.source_registry", blockers=blockers)
    families = [_mapping(value) for value in _sequence(track_a.get("families"))]
    family_uids = [str(row.get("family_uid")) for row in families]
    if len(family_uids) != 12 or set(family_uids) != ALLEN_FAMILY_UIDS or len(family_uids) != len(set(family_uids)):
        blockers.append(blocker("allen_family_inventory", "Allen family inventory must contain exactly 12 unique canonical families."))
    if any(not row.get("adapter_kind") or not _sequence(row.get("modalities")) for row in families):
        blockers.append(blocker("allen_family_contract", "Every Allen family requires an adapter kind and modality inventory."))
    migration = _mapping(track_a.get("historical_migration"))
    if migration.get("status") != "not_started" or migration.get("wholesale_copy_allowed") is not False:
        blockers.append(blocker("allen_migration_status", "Historical migration must remain unstarted and prohibit wholesale copying."))
    rights = _mapping(track_a.get("rights_matrix"))
    if set(_sequence(rights.get("required_fields"))) != RIGHTS_FIELDS or set(_sequence(rights.get("allowed_decisions"))) != RIGHTS_DECISIONS:
        blockers.append(blocker("allen_rights_contract", "Allen rights fields or decisions are not exact."))
    if rights.get("status") != "not_started" or rights.get("current_terms_snapshot_required") is not True:
        blockers.append(blocker("allen_rights_status", "Allen rights resolution is not correctly fail-closed."))
    release = _mapping(track_a.get("release_inventory"))
    if set(_sequence(release.get("required_fields"))) != RELEASE_FIELDS or release.get("mutable_latest_query_allowed") is not False:
        blockers.append(blocker("allen_release_contract", "Allen release identity contract is incomplete or permits mutable latest queries."))
    if set(_sequence(_mapping(track_a.get("adapter_contract")).get("required_methods"))) != ADAPTER_METHODS:
        blockers.append(blocker("allen_adapter_contract", "Allen adapter method inventory is not exact."))
    if set(_sequence(_mapping(track_a.get("parser_contract")).get("parser_families"))) != PARSER_FAMILIES:
        blockers.append(blocker("allen_parser_contract", "Allen parser family inventory is not exact."))
    if set(_sequence(track_a.get("deterministic_consumers"))) != DETERMINISTIC_CONSUMERS:
        blockers.append(blocker("allen_consumer_contract", "Allen deterministic consumer inventory is not exact."))
    authorization_a = _mapping(track_a.get("authorization"))
    if authorization_a.get("payload_download") is not False or authorization_a.get("training") is not False:
        blockers.append(blocker("allen_authorization", "Track A contract must not authorize payload download or training."))
    if not blockers:
        historical = historical_allen_summary(contract, repository)
        if historical != {
            "source_card_count": 12,
            "source_card_family_uids": sorted(ALLEN_FAMILY_UIDS),
            "source_cards_with_raw_hashes": 0,
            "source_cards_with_release_identity": 0,
            "historical_gate_status": "blocked_allen_license_or_terms",
            "historical_junit": {"tests": 598, "failures": 17, "errors": 19, "skipped": 0},
        }:
            blockers.append(blocker("historical_allen_drift", "Historical Allen zero-payload or failed-test evidence drifted.", observed=historical))

    track_b = _mapping(tracks["track_b_public_e2_e4_recovery"])
    _validate_hashed_inputs(track_b.get("reused_authorities"), repository=repository, label="track_b.reused_authorities", blockers=blockers)
    if set(_sequence(track_b.get("numeric_e2_required_fields"))) != E2_FIELDS:
        blockers.append(blocker("e2_fields", "Numeric E2 field inventory is not exact."))
    if set(_sequence(track_b.get("label_classes"))) != LABEL_CLASSES:
        blockers.append(blocker("e2_labels", "E2 label classes drifted from the Round-19 authority."))
    if set(_sequence(track_b.get("allowed_readiness_states"))) != READINESS_STATES:
        blockers.append(blocker("readiness_states", "Training-readiness states drifted from the Round-19 authority."))
    if track_b.get("current_readiness") != "blocked_insufficient_e4":
        blockers.append(blocker("readiness_promotion", "Track B cannot promote inherited blocked_insufficient_e4 before evidence recovery."))
    boundary = _mapping(track_b.get("evidence_boundary"))
    if boundary != {"simulation_is_model_derived": True, "model_prediction_closes_gap": False, "author_narrative_is_numeric_e2": False}:
        blockers.append(blocker("evidence_boundary", "Track B evidence-type boundary is not exact."))
    authorization_b = _mapping(track_b.get("authorization"))
    if authorization_b.get("intervention_control_pilot_execution") is not False or authorization_b.get("training") is not False:
        blockers.append(blocker("track_b_authorization", "Track B contract must not authorize pilots or training."))

    track_c = _mapping(tracks["track_c_runtime_deployment"])
    _validate_hashed_inputs(track_c.get("input_authorities"), repository=repository, label="track_c.input_authorities", blockers=blockers)
    ownership = _mapping(track_c.get("registry_ownership"))
    if set(ownership) != REGISTRY_NAMES:
        blockers.append(blocker("registry_inventory", "Runtime registry inventory is not exact."))
    if any(_mapping(value).get("owner") != "round20-shared-governance" for value in ownership.values()):
        blockers.append(blocker("registry_owner", "Every runtime registry must retain the shared owner."))
    if track_c.get("duplicate_track_specific_registries_allowed") is not False:
        blockers.append(blocker("duplicate_registry", "Track-specific duplicate registries are forbidden."))
    service = _mapping(track_c.get("service_contract"))
    required_endpoints = set(_sequence(service.get("required_endpoints")))
    existing_endpoints = set(_sequence(service.get("existing_exact_endpoints")))
    gaps = set(_sequence(service.get("implementation_gaps")))
    if required_endpoints != SERVICE_ENDPOINTS or not existing_endpoints <= required_endpoints or gaps != required_endpoints - existing_endpoints:
        blockers.append(blocker("service_endpoints", "Required, existing, and missing service endpoints do not reconcile."))
    if service.get("arbitrary_shell_execution_allowed") is not False or service.get("real_tool_dispatch_authorized") is not False:
        blockers.append(blocker("service_authorization", "Arbitrary shell execution and real tool dispatch must remain disabled."))
    authorization_c = _mapping(track_c.get("authorization"))
    if authorization_c.get("real_tool_dispatch") is not False or authorization_c.get("deployment") is not False:
        blockers.append(blocker("track_c_authorization", "Track C contract must not authorize dispatch or deployment."))
    return sorted_blockers(blockers)


def build_track_contract_report(
    contract: Mapping[str, object], *, repository: Path
) -> dict[str, object]:
    blockers = validate_round20_track_contracts(contract, repository=repository)
    report: dict[str, object] = {
        "schema_version": "neuronsifter-round20-track-contract-freeze-v1",
        "gate_uid": "round20_track_contract_freeze",
        "status": "passed" if not blockers else "blocked",
        "blocker_count": len(blockers),
        "blockers": blockers,
        "contract_hash": hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        ).hexdigest(),
        "base_main_commit": contract.get("base_main_commit"),
        "track_statuses": {
            name: _mapping(value).get("status")
            for name, value in sorted(_mapping(contract.get("tracks")).items())
        },
        "scope": {
            "strict_public_only_flag": True,
            "training_authorized": False,
            "training_started": False,
            "allen_payload_download_authorized": False,
            "real_tool_dispatch_authorized": False,
            "deployment_authorized": False,
            "evidence_fabric_sealed": False,
            "dt_core_status": "not_evaluated",
            "remote_push_authorized": False,
        },
        "next_action": (
            "execute_bounded_track_a_migration_rights_release_audits_track_b_source_discovery_and_track_c_registry_gap_implementation"
            if not blockers
            else "repair_track_contract_blockers_and_rerun_with_new_identity"
        ),
    }
    if not blockers:
        report["historical_allen"] = historical_allen_summary(contract, repository)
    return report
