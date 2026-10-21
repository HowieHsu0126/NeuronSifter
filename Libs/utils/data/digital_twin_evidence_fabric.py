from __future__ import annotations

import csv
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

from Libs.utils.data.raw_data_registry import (
    read_tsv,
    repo_root,
    resolve_repo_path,
    sha256_of_file,
    write_json,
    write_tsv,
)


REQUIRED_EVIDENCE_FIELDS = [
    "evidence_uid",
    "evidence_type",
    "source_uid",
    "publication_accession",
    "biological_unit_uid",
    "state_uid",
    "action_uid",
    "trajectory_uid",
    "tool_result_uid",
    "model_family_uid",
    "time_semantics",
    "pairing_semantics",
    "fidelity",
    "qc_grade",
    "uncertainty",
    "provenance",
    "allowed_train_roles",
    "allowed_eval_roles",
    "claim_eligibility",
]

REQUIRED_EVIDENCE_TYPES = {
    "real_static_state",
    "real_perturbation_endpoint",
    "real_longitudinal_trajectory",
    "legacy_simulator_trajectory",
    "clean_simulator_trajectory",
    "published_mechanistic_output",
    "tool_result",
    "hypothetical_scenario",
    "ontology_or_prior",
}

REQUIRED_PAIRING_SEMANTICS = {
    "same_unit_longitudinal",
    "same_unit_action_endpoint",
    "cross_sectional",
    "semantic_alignment_only",
    "simulator_generated",
    "not_paired",
}

LINK_FIELDS = (
    "biological_unit_uid",
    "state_uid",
    "action_uid",
    "trajectory_uid",
    "tool_result_uid",
    "model_family_uid",
)


def canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def stable_uid(prefix: str, payload: object) -> str:
    return f"{prefix}_{canonical_sha256(payload)[:24]}"


def evidence_uid_for_record(record: Mapping[str, object]) -> str:
    identity_payload = {field: deepcopy(record.get(field)) for field in REQUIRED_EVIDENCE_FIELDS if field != "evidence_uid"}
    return stable_uid("dt_evidence", identity_payload)


def _block(code: str, message: str, **details: object) -> dict[str, object]:
    return {"code": code, "message": message, **details}


def _sorted_blockers(blockers: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    unique = {canonical_json(dict(blocker)): dict(blocker) for blocker in blockers}
    return sorted(unique.values(), key=lambda row: (str(row.get("code", "")), str(row.get("path", "")), canonical_json(row)))


def _mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: object) -> list[object]:
    return list(value) if isinstance(value, list) else []


def fabric_config(config: Mapping[str, object]) -> dict[str, object]:
    payload = config.get("digital_twin_evidence_fabric")
    if not isinstance(payload, Mapping):
        raise KeyError("digital_twin_evidence_fabric")
    return dict(payload)


def validate_fabric_config(config: Mapping[str, object]) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    try:
        fabric = fabric_config(config)
    except KeyError:
        return [_block("missing_fabric_config", "Configuration is missing digital_twin_evidence_fabric.")]

    schemas = _mapping(config.get("schemas"))
    schema_keys = _mapping(fabric.get("schema_field_keys"))
    expected_schema_fields = {
        "evidence_record": REQUIRED_EVIDENCE_FIELDS,
        "relationship": [
            "edge_uid",
            "source_evidence_uid",
            "target_uid",
            "target_kind",
            "relationship_type",
            "pairing_semantics",
            "provenance",
        ],
        "role_matrix": [
            "fabric_identity",
            "evidence_uid",
            "evidence_type",
            "source_uid",
            "fidelity",
            "qc_grade",
            "allowed_train_roles",
            "allowed_eval_roles",
            "claim_eligibility",
        ],
    }
    for family, expected_fields in expected_schema_fields.items():
        field_key = str(schema_keys.get(family, ""))
        if not field_key or schemas.get(field_key) != expected_fields:
            blockers.append(
                _block(
                    "invalid_schema_field_owner",
                    "The DT evidence schema field order must be owned by schemas.yaml and match the contract.",
                    schema_family=family,
                    field_key=field_key,
                )
            )

    if str(fabric.get("version", "")) != "digital_twin_evidence_fabric_v1":
        blockers.append(_block("invalid_fabric_version", "The evidence fabric version must be digital_twin_evidence_fabric_v1."))
    for key in ("inventory", "relationship_graph", "validation"):
        if not str(_mapping(fabric.get("schema_versions")).get(key, "")):
            blockers.append(_block("missing_schema_version", "A DT evidence output schema version is missing.", output=key))
    if set(str(value) for value in _list(fabric.get("evidence_types"))) != REQUIRED_EVIDENCE_TYPES:
        blockers.append(_block("invalid_evidence_type_enum", "The evidence-type enum does not exactly match the v1 contract."))
    if set(str(value) for value in _list(fabric.get("pairing_semantics"))) != REQUIRED_PAIRING_SEMANTICS:
        blockers.append(_block("invalid_pairing_semantics_enum", "The pairing-semantics enum does not exactly match the v1 contract."))

    output_paths = _mapping(fabric.get("output_paths"))
    for output_name in ("inventory", "relationship_graph", "role_matrix", "validation"):
        path_value = str(output_paths.get(output_name, ""))
        if not path_value or Path(path_value).is_absolute():
            blockers.append(_block("invalid_output_path", "DT evidence output paths must be repository-relative.", output=output_name, path=path_value))

    supported_readers = {"json_artifact", "json_records", "tsv_records"}
    adapter_ids: set[str] = set()
    adapter_paths: set[str] = set()
    for index, raw_adapter in enumerate(_list(fabric.get("source_adapters"))):
        if not isinstance(raw_adapter, Mapping):
            blockers.append(_block("invalid_source_adapter", "Each source adapter must be a mapping.", index=index))
            continue
        adapter = dict(raw_adapter)
        adapter_id = str(adapter.get("adapter_id", ""))
        path_value = str(adapter.get("path", ""))
        if not adapter_id or adapter_id in adapter_ids:
            blockers.append(_block("duplicate_source_adapter", "Source adapter ids must be non-empty and unique.", adapter_id=adapter_id))
        adapter_ids.add(adapter_id)
        if not path_value or Path(path_value).is_absolute():
            blockers.append(_block("absolute_or_missing_source_path", "Source adapter paths must be non-empty and repository-relative.", adapter_id=adapter_id, path=path_value))
        if path_value in adapter_paths:
            blockers.append(_block("duplicate_source_path", "A frozen source path may have only one adapter owner.", path=path_value))
        adapter_paths.add(path_value)
        if str(adapter.get("reader_type", "")) not in supported_readers:
            blockers.append(_block("unsupported_reader_type", "Source adapter reader_type is unsupported.", adapter_id=adapter_id, reader_type=adapter.get("reader_type")))
        if not isinstance(adapter.get("required"), bool):
            blockers.append(_block("invalid_source_required_flag", "Source adapter required must be boolean.", adapter_id=adapter_id))
        defaults = _mapping(adapter.get("evidence_defaults"))
        missing_defaults = sorted(set(REQUIRED_EVIDENCE_FIELDS) - {"evidence_uid", "provenance"} - set(defaults))
        if missing_defaults:
            blockers.append(_block("missing_evidence_defaults", "Source adapter lacks canonical evidence defaults.", adapter_id=adapter_id, fields=missing_defaults))

    if not adapter_ids:
        blockers.append(_block("missing_source_adapters", "At least one read-only source adapter is required."))
    return _sorted_blockers(blockers)


def _safe_repo_path(root: Path, path_value: str) -> tuple[Path | None, dict[str, object] | None]:
    raw_path = Path(path_value)
    if raw_path.is_absolute():
        return None, _block("absolute_source_path", "Frozen input paths must be repository-relative.", path=path_value)
    resolved_root = root.resolve()
    candidate = resolved_root / raw_path
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        return None, _block("source_path_escape", "Frozen input path resolves outside the assigned repository.", path=path_value)
    return resolved, None


def _dotted_get(payload: object, dotted_path: str) -> object:
    current = payload
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(dotted_path)
        current = current[part]
    return current


def _json_records(payload: object, record_path: str) -> list[dict[str, object]]:
    selected = payload if not record_path else _dotted_get(payload, record_path)
    if not isinstance(selected, list) or any(not isinstance(row, Mapping) for row in selected):
        raise TypeError(f"JSON record path must resolve to a list of mappings: {record_path or '<root>'}")
    return [dict(row) for row in selected]


def _first_value(row: Mapping[str, object], candidates: object) -> object:
    for candidate in _list(candidates):
        value = row.get(str(candidate))
        if value is not None and value != "":
            return deepcopy(value)
    return None


def _record_from_source(
    adapter: Mapping[str, object],
    source_row: Mapping[str, object] | None,
    *,
    row_index: int | None,
    path_value: str,
    source_sha256: str,
    config: Mapping[str, object],
) -> dict[str, object]:
    defaults = deepcopy(_mapping(adapter.get("evidence_defaults")))
    record = {field: defaults.get(field) for field in REQUIRED_EVIDENCE_FIELDS if field != "evidence_uid"}
    row = dict(source_row or {})
    field_sources = _mapping(adapter.get("field_sources"))
    for field, candidates in field_sources.items():
        if field not in record or field == "provenance":
            continue
        value = _first_value(row, candidates)
        if value is not None:
            record[field] = value

    provenance = deepcopy(_mapping(defaults.get("provenance")))
    provenance.update(
        {
            "source_adapter_id": str(adapter.get("adapter_id", "")),
            "source_path": path_value,
            "source_sha256": source_sha256,
            "reader_type": str(adapter.get("reader_type", "")),
            "row_index": row_index,
            "source_record_sha256": canonical_sha256(row) if source_row is not None else source_sha256,
        }
    )
    selected_metadata = {
        str(field): deepcopy(row[str(field)])
        for field in _list(adapter.get("provenance_fields"))
        if str(field) in row
    }
    if selected_metadata:
        provenance["source_metadata"] = selected_metadata
    source_row_identity = {
        str(field): deepcopy(row[str(field)])
        for field in _list(adapter.get("uid_columns"))
        if str(field) in row and row[str(field)] not in (None, "")
    }
    if source_row_identity:
        provenance["source_row_identity"] = source_row_identity

    external_kinds = _mapping(fabric_config(config).get("external_entity_kinds"))
    external_entities = _mapping(provenance.get("external_entities"))
    for link_field in LINK_FIELDS:
        link_uid = record.get(link_field)
        if link_uid is not None and str(link_uid):
            external_entities[str(link_uid)] = str(external_kinds.get(link_field, ""))
    if external_entities:
        provenance["external_entities"] = dict(sorted(external_entities.items()))
    record["provenance"] = provenance
    record["evidence_uid"] = evidence_uid_for_record(record)
    return {field: record.get(field) for field in REQUIRED_EVIDENCE_FIELDS}


def _read_adapter_payload(path: Path, reader_type: str, record_path: str) -> tuple[object, list[dict[str, object]], object]:
    if reader_type in {"json_artifact", "json_records"}:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = _json_records(payload, record_path) if reader_type == "json_records" else []
        schema_or_header = _mapping(payload).get("schema_version")
        return payload, rows, schema_or_header
    if reader_type == "tsv_records":
        with path.open("r", encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle, delimiter="\t"), [])
        rows = [dict(row) for row in read_tsv(path)]
        return rows, rows, header
    raise ValueError(f"Unsupported reader_type: {reader_type}")


def _identity_observations(payload: object, expected_identity: Mapping[str, object]) -> tuple[dict[str, object], bool]:
    observed: dict[str, object] = {}
    matches = True
    for field, expected in expected_identity.items():
        try:
            value = _dotted_get(payload, str(field))
        except KeyError:
            value = None
        observed[str(field)] = value
        if value != expected:
            matches = False
    return observed, matches


def _audit_source_adapter(
    config: Mapping[str, object],
    adapter: Mapping[str, object],
    root: Path,
    *,
    hash_function: Callable[[Path], str] = sha256_of_file,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    adapter_id = str(adapter.get("adapter_id", ""))
    path_value = str(adapter.get("path", ""))
    required = bool(adapter.get("required", False))
    reader_type = str(adapter.get("reader_type", ""))
    input_audit: dict[str, object] = {
        "adapter_id": adapter_id,
        "path": path_value,
        "required": required,
        "reader_type": reader_type,
        "exists": False,
        "status": "not_read",
        "size_bytes": None,
        "sha256_before": None,
        "sha256_after": None,
        "schema_version_or_header": None,
        "expected_identity": deepcopy(_mapping(adapter.get("expected_identity"))),
        "observed_identity": {},
        "identity_verified": False,
        "record_count": 0,
        "evidence_uids": [],
    }
    blockers: list[dict[str, object]] = []
    source_path, path_blocker = _safe_repo_path(root, path_value)
    if path_blocker is not None:
        blockers.append({**path_blocker, "adapter_id": adapter_id})
        input_audit["status"] = "invalid_path"
        return input_audit, [], blockers
    assert source_path is not None
    if not source_path.is_file():
        input_audit["status"] = "missing_required" if required else "missing_optional"
        if required:
            blockers.append(
                _block(
                    "missing_required_frozen_input",
                    "A required frozen input is absent from the assigned repository snapshot.",
                    adapter_id=adapter_id,
                    path=path_value,
                )
            )
        return input_audit, [], blockers

    input_audit["exists"] = True
    input_audit["size_bytes"] = source_path.stat().st_size
    try:
        before_hash = hash_function(source_path)
        input_audit["sha256_before"] = before_hash
        payload, rows, schema_or_header = _read_adapter_payload(source_path, reader_type, str(adapter.get("record_path", "")))
        input_audit["schema_version_or_header"] = schema_or_header
        after_hash = hash_function(source_path)
        input_audit["sha256_after"] = after_hash
    except (OSError, ValueError, TypeError, json.JSONDecodeError, csv.Error) as exc:
        input_audit["status"] = "malformed"
        blockers.append(_block("malformed_frozen_input", "A frozen input could not be parsed through its declared reader.", adapter_id=adapter_id, path=path_value, error=str(exc)))
        return input_audit, [], blockers

    if before_hash != after_hash:
        input_audit["status"] = "mutated_during_read"
        blockers.append(_block("input_mutated_during_read", "A frozen input hash changed while it was being read.", adapter_id=adapter_id, path=path_value, sha256_before=before_hash, sha256_after=after_hash))
        return input_audit, [], blockers

    expected_identity = _mapping(adapter.get("expected_identity"))
    observed_identity, identity_matches = _identity_observations(payload, expected_identity)
    input_audit["observed_identity"] = observed_identity
    input_audit["identity_verified"] = identity_matches
    if not identity_matches:
        input_audit["status"] = "identity_mismatch"
        blockers.append(_block("frozen_input_identity_mismatch", "A frozen input does not match its configured identity.", adapter_id=adapter_id, path=path_value, expected=expected_identity, observed=observed_identity))
        return input_audit, [], blockers

    if reader_type == "tsv_records":
        header = set(str(value) for value in _list(input_audit["schema_version_or_header"]))
        required_columns = set(str(value) for value in _list(adapter.get("required_columns")))
        missing_columns = sorted(required_columns - header)
        if missing_columns:
            input_audit["status"] = "schema_mismatch"
            blockers.append(_block("missing_required_source_columns", "A TSV source lacks required configured columns.", adapter_id=adapter_id, path=path_value, columns=missing_columns))
            return input_audit, [], blockers

    required_row_match = _mapping(adapter.get("required_row_match"))
    if required_row_match:
        field = str(required_row_match.get("field", ""))
        value = required_row_match.get("value")
        if not any(row.get(field) == value for row in rows):
            blockers.append(_block("missing_required_source_holding", "A configured source holding is absent from its provenance-bearing registry.", adapter_id=adapter_id, path=path_value, field=field, value=value))

    uid_columns = [str(value) for value in _list(adapter.get("uid_columns"))]
    for row_index, row in enumerate(rows):
        missing_uid_columns = [field for field in uid_columns if row.get(field) in (None, "")]
        if missing_uid_columns:
            blockers.append(
                _block(
                    "missing_source_row_identity",
                    "A source record lacks its configured stable identity fields.",
                    adapter_id=adapter_id,
                    path=path_value,
                    row_index=row_index,
                    fields=missing_uid_columns,
                )
            )

    if reader_type == "json_artifact":
        records = [_record_from_source(adapter, None, row_index=None, path_value=path_value, source_sha256=before_hash, config=config)]
    else:
        records = [
            _record_from_source(adapter, row, row_index=index, path_value=path_value, source_sha256=before_hash, config=config)
            for index, row in enumerate(rows)
        ]
    input_audit["record_count"] = len(records)
    input_audit["evidence_uids"] = sorted(str(record["evidence_uid"]) for record in records)
    input_audit["status"] = "read_ok" if not blockers else "read_with_blockers"
    return input_audit, records, blockers


def validate_evidence_records(config: Mapping[str, object], records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    fabric = fabric_config(config)
    evidence_types = set(str(value) for value in _list(fabric.get("evidence_types")))
    pairing_values = set(str(value) for value in _list(fabric.get("pairing_semantics")))
    time_values = set(str(value) for value in _list(fabric.get("time_semantics")))
    fidelities = set(str(value) for value in _list(fabric.get("fidelity_levels")))
    qc_grades = set(str(value) for value in _list(fabric.get("qc_grades")))
    train_roles = set(str(value) for value in _list(fabric.get("allowed_train_roles")))
    eval_roles = set(str(value) for value in _list(fabric.get("allowed_eval_roles")))
    claim_values = set(str(value) for value in _list(fabric.get("claim_eligibility_values")))
    role_policy = _mapping(fabric.get("role_policy"))
    generated_types = set(str(value) for value in _list(role_policy.get("generated_evidence_types")))
    blockers: list[dict[str, object]] = []
    content_by_uid: dict[str, str] = {}

    for index, record_value in enumerate(records):
        record = dict(record_value)
        uid = str(record.get("evidence_uid", ""))
        if list(record.keys()) != REQUIRED_EVIDENCE_FIELDS:
            blockers.append(_block("invalid_evidence_field_order", "Evidence records must contain exactly the canonical ordered fields.", index=index, evidence_uid=uid, fields=list(record.keys())))
            continue
        if not uid or uid != evidence_uid_for_record(record):
            blockers.append(_block("invalid_evidence_uid", "Evidence UID does not match canonical record content.", index=index, evidence_uid=uid))
        canonical_record = canonical_json(record)
        if uid in content_by_uid and content_by_uid[uid] != canonical_record:
            blockers.append(_block("conflicting_duplicate_evidence_uid", "One evidence UID resolves to unequal record content.", evidence_uid=uid))
        content_by_uid[uid] = canonical_record
        evidence_type = str(record.get("evidence_type", ""))
        pairing = str(record.get("pairing_semantics", ""))
        time_semantics = str(record.get("time_semantics", ""))
        if evidence_type not in evidence_types:
            blockers.append(_block("invalid_evidence_type", "Evidence record uses an undeclared evidence_type.", evidence_uid=uid, value=evidence_type))
        if pairing not in pairing_values:
            blockers.append(_block("invalid_pairing_semantics", "Evidence record uses an undeclared pairing_semantics.", evidence_uid=uid, value=pairing))
        if time_semantics not in time_values:
            blockers.append(_block("invalid_time_semantics", "Evidence record uses an undeclared time_semantics.", evidence_uid=uid, value=time_semantics))
        if str(record.get("fidelity", "")) not in fidelities:
            blockers.append(_block("invalid_fidelity", "Evidence record uses an undeclared fidelity.", evidence_uid=uid, value=record.get("fidelity")))
        if str(record.get("qc_grade", "")) not in qc_grades:
            blockers.append(_block("invalid_qc_grade", "Evidence record uses an undeclared QC grade.", evidence_uid=uid, value=record.get("qc_grade")))
        if str(record.get("claim_eligibility", "")) not in claim_values:
            blockers.append(_block("invalid_claim_eligibility", "Evidence record uses an undeclared claim eligibility.", evidence_uid=uid, value=record.get("claim_eligibility")))
        if not str(record.get("source_uid", "")):
            blockers.append(_block("missing_source_uid", "Evidence records require a non-empty source_uid.", evidence_uid=uid))
        if not isinstance(record.get("uncertainty"), Mapping) or not record.get("uncertainty"):
            blockers.append(_block("invalid_uncertainty", "Evidence uncertainty must be a non-empty mapping.", evidence_uid=uid))
        if not isinstance(record.get("provenance"), Mapping) or not record.get("provenance"):
            blockers.append(_block("invalid_provenance", "Evidence provenance must be a non-empty mapping.", evidence_uid=uid))
        record_train_roles = record.get("allowed_train_roles")
        record_eval_roles = record.get("allowed_eval_roles")
        if not isinstance(record_train_roles, list) or any(str(value) not in train_roles for value in record_train_roles):
            blockers.append(_block("invalid_train_role", "Evidence train roles must be a list of configured values.", evidence_uid=uid, roles=record_train_roles))
            record_train_roles = []
        if not isinstance(record_eval_roles, list) or any(str(value) not in eval_roles for value in record_eval_roles):
            blockers.append(_block("invalid_eval_role", "Evidence evaluation roles must be a list of configured values.", evidence_uid=uid, roles=record_eval_roles))
            record_eval_roles = []
        if len(record_train_roles) != len(set(str(value) for value in record_train_roles)) or len(record_eval_roles) != len(set(str(value) for value in record_eval_roles)):
            blockers.append(_block("duplicate_role", "Evidence role lists cannot contain duplicates.", evidence_uid=uid))

        if evidence_type in {"real_static_state", "real_perturbation_endpoint"} and (time_semantics == "longitudinal" or pairing == "same_unit_longitudinal"):
            blockers.append(_block("static_evidence_marked_longitudinal", "Static states and perturbation endpoints cannot be labeled longitudinal.", evidence_uid=uid))
        if evidence_type == "real_longitudinal_trajectory" and (time_semantics != "longitudinal" or pairing != "same_unit_longitudinal"):
            blockers.append(_block("invalid_longitudinal_contract", "Real longitudinal evidence requires longitudinal time and same-unit pairing.", evidence_uid=uid))
        if evidence_type == "real_perturbation_endpoint" and time_semantics != "endpoint":
            blockers.append(_block("invalid_endpoint_time_semantics", "Real perturbation endpoints require endpoint time semantics.", evidence_uid=uid))
        if evidence_type in generated_types and ({"state_reconstruction", "endpoint_calibration"} & set(str(value) for value in record_train_roles)):
            blockers.append(_block("generated_evidence_as_observation", "Generated/model/tool evidence cannot satisfy real-observation training roles.", evidence_uid=uid))
        if evidence_type == "legacy_simulator_trajectory":
            forbidden_train = set(str(value) for value in _list(role_policy.get("legacy_forbidden_train_roles")))
            forbidden_eval = set(str(value) for value in _list(role_policy.get("legacy_forbidden_eval_roles")))
            if forbidden_train & set(str(value) for value in record_train_roles) or forbidden_eval & set(str(value) for value in record_eval_roles):
                blockers.append(_block("legacy_physical_action_role", "Legacy trajectories cannot support physical action, dose, route/schedule, action-contrast, or strict-action-holdout roles.", evidence_uid=uid))
        provenance = _mapping(record.get("provenance"))
        if provenance.get("qualification_status") == "unqualified_non_core":
            allowed_non_core_train = set(str(value) for value in _list(role_policy.get("unqualified_non_core_train_roles")))
            allowed_non_core_eval = set(str(value) for value in _list(role_policy.get("unqualified_non_core_eval_roles")))
            if not set(str(value) for value in record_train_roles).issubset(allowed_non_core_train) or not set(str(value) for value in record_eval_roles).issubset(allowed_non_core_eval):
                blockers.append(_block("invalid_unqualified_non_core_role", "Unqualified non-core evidence is limited to context_prior, weak_calibration, or holdout roles.", evidence_uid=uid))
        if record.get("model_family_uid") is not None and provenance.get("model_family_uid") not in (None, record.get("model_family_uid")):
            blockers.append(_block("contradictory_model_ownership", "Evidence model-family ownership conflicts with provenance.", evidence_uid=uid))
    return _sorted_blockers(blockers)


def build_relationship_graph(config: Mapping[str, object], records: Sequence[Mapping[str, object]], fabric_identity: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    fabric = fabric_config(config)
    relationship_types = _mapping(fabric.get("relationship_types"))
    external_kinds_by_field = _mapping(fabric.get("external_entity_kinds"))
    record_uids = {str(record["evidence_uid"]) for record in records}
    external_nodes: dict[str, str] = {}
    edges: list[dict[str, object]] = []
    blockers: list[dict[str, object]] = []

    for record_value in records:
        record = dict(record_value)
        source_uid = str(record["evidence_uid"])
        provenance = _mapping(record.get("provenance"))
        declared_external = {str(key): str(value) for key, value in _mapping(provenance.get("external_entities")).items()}
        for link_field in LINK_FIELDS:
            target_value = record.get(link_field)
            if target_value is None or not str(target_value):
                continue
            target_uid = str(target_value)
            if target_uid == source_uid:
                blockers.append(_block("self_relationship", "Evidence cannot claim a relationship to itself.", evidence_uid=source_uid, field=link_field))
                continue
            if target_uid in record_uids:
                target_kind = "evidence"
            else:
                target_kind = declared_external.get(target_uid, "")
                expected_kind = str(external_kinds_by_field.get(link_field, ""))
                if not target_kind or target_kind != expected_kind:
                    blockers.append(_block("dangling_relationship_target", "Relationship targets must resolve to evidence or an explicitly typed external entity.", evidence_uid=source_uid, field=link_field, target_uid=target_uid, expected_kind=expected_kind))
                    continue
                external_nodes[target_uid] = target_kind
            relationship_type = str(relationship_types.get(link_field, ""))
            edge_payload = {
                "source_evidence_uid": source_uid,
                "target_uid": target_uid,
                "target_kind": target_kind,
                "relationship_type": relationship_type,
                "pairing_semantics": str(record["pairing_semantics"]),
                "provenance": {"source_field": link_field},
            }
            edges.append({"edge_uid": stable_uid("dt_edge", edge_payload), **edge_payload})

        for relationship in _list(provenance.get("relationships")):
            if not isinstance(relationship, Mapping):
                blockers.append(_block("invalid_relationship_declaration", "Provenance relationships must be mappings.", evidence_uid=source_uid))
                continue
            target_uid = str(relationship.get("target_uid", ""))
            relationship_type = str(relationship.get("relationship_type", ""))
            if target_uid == source_uid:
                blockers.append(_block("self_relationship", "Evidence cannot claim a relationship to itself.", evidence_uid=source_uid))
                continue
            if target_uid in record_uids:
                target_kind = "evidence"
            else:
                target_kind = declared_external.get(target_uid, "")
                if target_kind:
                    external_nodes[target_uid] = target_kind
            if not target_uid or (target_uid not in record_uids and not target_kind):
                blockers.append(_block("dangling_relationship_target", "Declared relationship target does not resolve.", evidence_uid=source_uid, target_uid=target_uid))
                continue
            allowed_relationships = set(str(value) for value in relationship_types.values())
            if relationship_type not in allowed_relationships:
                blockers.append(_block("invalid_relationship_type", "Declared relationship type is not configured.", evidence_uid=source_uid, relationship_type=relationship_type))
                continue
            pairing = str(record["pairing_semantics"])
            if relationship_type == relationship_types.get("paired_evidence") and pairing not in {"same_unit_longitudinal", "same_unit_action_endpoint"}:
                blockers.append(_block("invalid_pairing_edge", "A paired_with edge requires a same-unit pairing semantic.", evidence_uid=source_uid, pairing_semantics=pairing))
                continue
            if pairing == "semantic_alignment_only" and relationship_type != relationship_types.get("semantic_alignment"):
                blockers.append(_block("semantic_alignment_as_pairing", "Semantic alignment cannot be represented as a pairing relationship.", evidence_uid=source_uid, relationship_type=relationship_type))
                continue
            edge_payload = {
                "source_evidence_uid": source_uid,
                "target_uid": target_uid,
                "target_kind": target_kind,
                "relationship_type": relationship_type,
                "pairing_semantics": pairing,
                "provenance": {"source_field": "provenance.relationships"},
            }
            edges.append({"edge_uid": stable_uid("dt_edge", edge_payload), **edge_payload})

    edge_fields = _mapping(config.get("schemas")).get(str(_mapping(fabric.get("schema_field_keys")).get("relationship")), [])
    normalized_edges = [{str(field): edge.get(str(field)) for field in edge_fields} for edge in edges]
    normalized_edges.sort(key=lambda row: str(row["edge_uid"]))
    graph = {
        "schema_version": str(_mapping(fabric.get("schema_versions")).get("relationship_graph", "")),
        "fabric_version": str(fabric.get("version", "")),
        "fabric_identity": fabric_identity,
        "nodes": sorted(
            (
                {
                    "evidence_uid": str(record["evidence_uid"]),
                    "evidence_type": str(record["evidence_type"]),
                    "source_uid": str(record["source_uid"]),
                }
                for record in records
            ),
            key=lambda row: row["evidence_uid"],
        ),
        "external_nodes": [
            {"uid": uid, "kind": kind}
            for uid, kind in sorted(external_nodes.items())
        ],
        "edges": normalized_edges,
        "node_count": len(records),
        "external_node_count": len(external_nodes),
        "edge_count": len(normalized_edges),
        "blockers": _sorted_blockers(blockers),
        "blocker_count": len(_sorted_blockers(blockers)),
    }
    return graph, _sorted_blockers(blockers)


def build_role_matrix(config: Mapping[str, object], records: Sequence[Mapping[str, object]], fabric_identity: str) -> list[dict[str, object]]:
    fields = _mapping(config.get("schemas"))[str(_mapping(fabric_config(config).get("schema_field_keys")).get("role_matrix"))]
    rows: list[dict[str, object]] = []
    for record in records:
        row = {
            "fabric_identity": fabric_identity,
            "evidence_uid": record["evidence_uid"],
            "evidence_type": record["evidence_type"],
            "source_uid": record["source_uid"],
            "fidelity": record["fidelity"],
            "qc_grade": record["qc_grade"],
            "allowed_train_roles": canonical_json(record["allowed_train_roles"]),
            "allowed_eval_roles": canonical_json(record["allowed_eval_roles"]),
            "claim_eligibility": record["claim_eligibility"],
        }
        rows.append({str(field): row[str(field)] for field in fields})
    return sorted(rows, key=lambda row: str(row["evidence_uid"]))


def _record_counts(config: Mapping[str, object], records: Sequence[Mapping[str, object]]) -> dict[str, object]:
    role_policy = _mapping(fabric_config(config).get("role_policy"))
    observation_types = set(str(value) for value in _list(role_policy.get("observation_evidence_types")))
    type_counts: dict[str, int] = {}
    train_role_counts: dict[str, int] = {}
    eval_role_counts: dict[str, int] = {}
    claim_counts: dict[str, int] = {}
    for record in records:
        evidence_type = str(record["evidence_type"])
        type_counts[evidence_type] = type_counts.get(evidence_type, 0) + 1
        for role in record["allowed_train_roles"]:
            train_role_counts[str(role)] = train_role_counts.get(str(role), 0) + 1
        for role in record["allowed_eval_roles"]:
            eval_role_counts[str(role)] = eval_role_counts.get(str(role), 0) + 1
        claim = str(record["claim_eligibility"])
        claim_counts[claim] = claim_counts.get(claim, 0) + 1
    return {
        "record_count": len(records),
        "observation_record_count": sum(1 for record in records if str(record["evidence_type"]) in observation_types),
        "evidence_type_counts": dict(sorted(type_counts.items())),
        "train_role_counts": dict(sorted(train_role_counts.items())),
        "eval_role_counts": dict(sorted(eval_role_counts.items())),
        "claim_eligibility_counts": dict(sorted(claim_counts.items())),
    }


def build_evidence_fabric_candidate(
    config: Mapping[str, object],
    *,
    repository_root: Path | None = None,
    hash_function: Callable[[Path], str] = sha256_of_file,
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    root = (repository_root or repo_root()).resolve()
    fabric = fabric_config(config)
    blockers = validate_fabric_config(config)
    input_artifacts: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    if not any(blocker["code"] in {"missing_fabric_config", "invalid_schema_field_owner"} for blocker in blockers):
        for raw_adapter in _list(fabric.get("source_adapters")):
            if not isinstance(raw_adapter, Mapping):
                continue
            input_audit, source_records, source_blockers = _audit_source_adapter(config, raw_adapter, root, hash_function=hash_function)
            input_artifacts.append(input_audit)
            records.extend(source_records)
            blockers.extend(source_blockers)

    records.sort(key=lambda row: str(row.get("evidence_uid", "")))
    deduplicated_records: list[dict[str, object]] = []
    seen_canonical: set[str] = set()
    for record in records:
        serialized = canonical_json(record)
        if serialized not in seen_canonical:
            seen_canonical.add(serialized)
            deduplicated_records.append(record)
    records = deduplicated_records
    blockers.extend(validate_evidence_records(config, records))
    input_artifacts.sort(key=lambda row: str(row["path"]))
    fabric_identity = stable_uid(
        "dt_fabric",
        {
            "version": fabric.get("version"),
            "input_hashes": [
                {"path": row["path"], "status": row["status"], "sha256": row["sha256_after"]}
                for row in input_artifacts
            ],
            "records": records,
        },
    )
    graph, graph_blockers = build_relationship_graph(config, records, fabric_identity)
    blockers.extend(graph_blockers)
    blockers = _sorted_blockers(blockers)
    counts = _record_counts(config, records)
    missing_only = bool(blockers) and all(blocker["code"] == "missing_required_frozen_input" for blocker in blockers)
    inventory = {
        "schema_version": str(_mapping(fabric.get("schema_versions")).get("inventory", "")),
        "fabric_version": str(fabric.get("version", "")),
        "fabric_identity": fabric_identity,
        "status": "candidate_complete" if not blockers else ("unsealed_missing_required_inputs" if missing_only else "unsealed_invalid"),
        "sealed": False,
        "repository_relative_inputs_only": True,
        "input_artifacts": input_artifacts,
        "input_artifact_count": len(input_artifacts),
        "records": records,
        **counts,
        "blockers": blockers,
        "blocker_count": len(blockers),
    }
    graph["blockers"] = blockers
    graph["blocker_count"] = len(blockers)
    role_rows = build_role_matrix(config, records, fabric_identity)
    return inventory, graph, role_rows


def _configured_output_path(config: Mapping[str, object], output_name: str, root: Path) -> Path:
    path_value = str(_mapping(fabric_config(config).get("output_paths"))[output_name])
    safe_path, blocker = _safe_repo_path(root, path_value)
    if blocker is not None or safe_path is None:
        raise ValueError(blocker)
    if root.resolve() == repo_root().resolve():
        return resolve_repo_path(path_value)
    return safe_path


def write_evidence_fabric_outputs(
    config: Mapping[str, object],
    inventory: Mapping[str, object],
    graph: Mapping[str, object],
    role_rows: Sequence[Mapping[str, object]],
    *,
    repository_root: Path | None = None,
) -> dict[str, Path]:
    root = (repository_root or repo_root()).resolve()
    paths = {
        name: _configured_output_path(config, name, root)
        for name in ("inventory", "relationship_graph", "role_matrix")
    }
    write_json(paths["inventory"], inventory)
    write_json(paths["relationship_graph"], graph)
    role_fields = _mapping(config.get("schemas"))[str(_mapping(fabric_config(config).get("schema_field_keys")).get("role_matrix"))]
    write_tsv(paths["role_matrix"], [str(field) for field in role_fields], [dict(row) for row in role_rows])
    return paths


def _load_role_matrix(config: Mapping[str, object], path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    blockers: list[dict[str, object]] = []
    fields = [str(field) for field in _mapping(config.get("schemas"))[str(_mapping(fabric_config(config).get("schema_field_keys")).get("role_matrix"))]]
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle, delimiter="\t"), [])
        if header != fields:
            blockers.append(_block("role_matrix_header_mismatch", "Role matrix header does not match schemas.yaml.", expected=fields, actual=header))
            return [], blockers
        raw_rows = read_tsv(path)
        rows: list[dict[str, object]] = []
        for raw_row in raw_rows:
            row: dict[str, object] = dict(raw_row)
            train_roles = json.loads(str(row["allowed_train_roles"]))
            eval_roles = json.loads(str(row["allowed_eval_roles"]))
            if not isinstance(train_roles, list) or not isinstance(eval_roles, list):
                raise TypeError("Role matrix role columns must encode JSON lists.")
            rows.append(row)
        return rows, blockers
    except (OSError, ValueError, TypeError, json.JSONDecodeError, csv.Error) as exc:
        blockers.append(_block("malformed_role_matrix", "Role matrix could not be read back.", error=str(exc)))
        return [], blockers


def validate_evidence_fabric_outputs(
    config: Mapping[str, object],
    *,
    repository_root: Path | None = None,
) -> dict[str, object]:
    root = (repository_root or repo_root()).resolve()
    fabric = fabric_config(config)
    output_paths = {
        name: _configured_output_path(config, name, root)
        for name in ("inventory", "relationship_graph", "role_matrix")
    }
    blockers: list[dict[str, object]] = []
    for name, path in output_paths.items():
        if not path.is_file():
            blockers.append(_block("missing_evidence_fabric_output", "A required evidence-fabric output is absent.", output=name, path=str(_mapping(fabric.get("output_paths"))[name])))
    inventory: dict[str, object] = {}
    graph: dict[str, object] = {}
    role_rows: list[dict[str, object]] = []
    if not blockers:
        try:
            with output_paths["inventory"].open("r", encoding="utf-8") as handle:
                inventory = json.load(handle)
            with output_paths["relationship_graph"].open("r", encoding="utf-8") as handle:
                graph = json.load(handle)
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            blockers.append(_block("malformed_evidence_fabric_output", "Evidence-fabric JSON output could not be read back.", error=str(exc)))
        role_rows, role_blockers = _load_role_matrix(config, output_paths["role_matrix"])
        blockers.extend(role_blockers)

    expected_inventory, expected_graph, expected_role_rows = build_evidence_fabric_candidate(config, repository_root=root)
    if inventory:
        if bool(inventory.get("sealed")):
            blockers.append(_block("blocked_output_falsely_labeled_sealed", "The builder inventory cannot label itself sealed before independent validation."))
        if inventory != expected_inventory:
            blockers.append(_block("inventory_readback_mismatch", "Inventory does not match independent source readback and canonical reconstruction."))
    if graph and graph != expected_graph:
        blockers.append(_block("relationship_graph_readback_mismatch", "Relationship graph does not match independent canonical reconstruction."))
    if role_rows != expected_role_rows:
        blockers.append(_block("role_matrix_readback_mismatch", "Role matrix does not contain exactly one canonical row per evidence record."))

    expected_blockers = _list(expected_inventory.get("blockers"))
    blockers.extend(dict(blocker) for blocker in expected_blockers if isinstance(blocker, Mapping))
    blockers = _sorted_blockers(blockers)
    missing_only = bool(blockers) and all(blocker["code"] == "missing_required_frozen_input" for blocker in blockers)
    sealing_policy = _mapping(fabric.get("sealing_policy"))
    if not blockers:
        status = str(sealing_policy.get("passed_status"))
    elif missing_only:
        status = str(sealing_policy.get("missing_status"))
    else:
        status = str(sealing_policy.get("invalid_status"))
    artifact_hashes = {
        name: sha256_of_file(path) if path.is_file() else None
        for name, path in output_paths.items()
    }
    return {
        "schema_version": str(_mapping(fabric.get("schema_versions")).get("validation", "")),
        "fabric_version": str(fabric.get("version", "")),
        "fabric_identity": expected_inventory.get("fabric_identity"),
        "status": status,
        "sealed": status == sealing_policy.get("passed_status"),
        "artifact_hashes": artifact_hashes,
        "record_count": expected_inventory.get("record_count", 0),
        "observation_record_count": expected_inventory.get("observation_record_count", 0),
        "relationship_node_count": expected_graph.get("node_count", 0),
        "relationship_edge_count": expected_graph.get("edge_count", 0),
        "role_matrix_row_count": len(expected_role_rows),
        "blockers": blockers,
        "blocker_count": len(blockers),
    }


def write_validation_report(
    config: Mapping[str, object],
    report: Mapping[str, object],
    *,
    repository_root: Path | None = None,
) -> Path:
    root = (repository_root or repo_root()).resolve()
    path = _configured_output_path(config, "validation", root)
    write_json(path, report)
    return path
