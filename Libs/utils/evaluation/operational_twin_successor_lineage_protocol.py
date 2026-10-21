"""Freeze and authenticate a physically isolated Operational Twin successor lineage."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Mapping, Sequence

from Libs.scripts.builders.build_operational_twin_development_training_commands import (
    build_candidate_commands,
)
from Libs.utils.evaluation.operational_twin_training_protocol import (
    CANDIDATE_COMMAND_ROW_KEYS,
    CANDIDATE_FORBIDDEN_STATE_INPUT_FIELDS,
    REQUIRED_METHOD_IDS,
    OperationalTwinProtocolError,
    build_candidate_assignments,
    canonical_hash,
    command_catalog_hash,
    expected_candidate_command_argv,
    read_repo_bytes,
    read_repo_json,
)


CONFIG_KEY = "operational_twin_successor_lineage_protocol"
SCHEMA_KEY = "operational_twin_successor_lineage_protocol_schema"
POSITIVE_TERMINAL = "OPERATIONAL_TWIN_SUCCESSOR_LINEAGE_PROTOCOL_FROZEN"
INVALID_TERMINAL = "OPERATIONAL_TWIN_SUCCESSOR_LINEAGE_PROTOCOL_INVALID"
MANIFEST_SCHEMA_VERSION = "neuronsifter-operational-twin-successor-lineage-manifest-v1"
ASSIGNMENT_SCHEMA_VERSION = (
    "neuronsifter-operational-twin-successor-assignment-manifest-v1"
)
COMMAND_SCHEMA_VERSION = "neuronsifter-operational-twin-successor-command-catalog-v1"
PREFLIGHT_SCHEMA_VERSION = (
    "neuronsifter-operational-twin-successor-lineage-preflight-v1"
)
AUDIT_SCHEMA_VERSION = "neuronsifter-operational-twin-successor-lineage-audit-v1"
PRE_CHANGE_SCHEMA_VERSION = (
    "neuronsifter-operational-twin-successor-pre-change-identity-manifest-v1"
)
SOURCE_GATE_SCHEMA_VERSION = (
    "neuronsifter-operational-twin-successor-source-gate-v1"
)
PACKAGE_GATE_SCHEMA_VERSION = (
    "neuronsifter-operational-twin-successor-reviewed-package-v1"
)
EXPECTED_AUTHORITATIVE_DATE = "2026-09-13"
EXPECTED_EVIDENCE_CLASS = "controlled_world_non_empirical_successor_lineage"
EXPECTED_PREDECESSOR_LINEAGE_UID = (
    "operational_twin_candidate_lineage_ef80556d38c9e1b50c28353a"
)
EXPECTED_PREDECESSOR_PROTOCOL_UID = (
    "operational_twin_candidate_protocol_d2a92a44cbb4a23b6fc91645"
)
EXPECTED_PREDECESSOR_PREFLIGHT_UID = (
    "operational_twin_candidate_lineage_preflight_1b0203b9807dd9f83876397b"
)
EXPECTED_SCIENTIFIC_CONTRACT_HASH = (
    "36ba40826b432170096191c8361931fe82c01b24c3d517939a541fa20a8d15b2"
)
EXPECTED_ROUND_48_DECISION_UID = (
    "operational_twin_candidate_retry_authority_05015fd1c0941e7e931cb47d"
)
EXPECTED_ROUND_48_TERMINAL = "OPERATIONAL_TWIN_SUCCESSOR_LINEAGE_PROTOCOL_REQUIRED"
EXPECTED_SEEDS = [360101, 360211, 360307]
EXPECTED_ATTEMPT_KINDS = ["initial", "infrastructure_recovery_01"]
EXPECTED_SOURCE_GATE_CHECKS = [
    "changed_file_ruff",
    "config_loading",
    "focused_tests",
    "git_diff_check",
    "source_predecessor_identity",
    "successor_adjudication",
]
PUBLICATION_KEYS = (
    "candidate_lineage_manifest",
    "assignment_manifest",
    "assignment_commands",
    "preflight_authority",
    "protocol_audit",
    "preflight_report",
    "protocol_report",
)
JSON_PUBLICATION_KEYS = PUBLICATION_KEYS[:5]
REVIEW_RECORD_KEYS = {
    "schema_version",
    "round_id",
    "roadmap_id",
    "roadmap_revision",
    "roadmap_dir",
    "milestone_id",
    "direction_id",
    "extracted_item_id",
    "decision",
    "evidence_summary",
    "roadmap_closeout",
}
SUCCESSOR_COMMAND_CATALOG_KEYS = {
    "schema_version",
    "candidate_lineage_uid",
    "protocol_uid",
    "preflight_uid",
    "candidate_manifest_hash",
    "assignment_manifest_hash",
    "assignment_count",
    "command_count",
    "initial_command_count",
    "conditional_recovery_command_count",
    "commands",
    "training_executed",
    "calibration_fitting_executed",
    "checkpoint_created",
    "assignment_bijection_hash",
    "attempt_slot_bijection_hash",
    "execution_handoff_required",
    "candidate_dispatch_authorized",
    "command_catalog_hash",
}


def _block(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _contract(config: Mapping[str, object]) -> dict[str, object]:
    value = config.get(CONFIG_KEY)
    if not isinstance(value, Mapping):
        raise OperationalTwinProtocolError(
            f"missing_successor_protocol_config: {CONFIG_KEY} is absent"
        )
    return dict(value)


def _schema(config: Mapping[str, object]) -> dict[str, object]:
    value = config.get(SCHEMA_KEY)
    if not isinstance(value, Mapping):
        raise OperationalTwinProtocolError(
            f"missing_successor_protocol_schema: {SCHEMA_KEY} is absent"
        )
    return dict(value)


def _git_object(root: Path, ref: str, path: str) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "--verify", f"{ref}:{path}"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _git_bytes(root: Path, ref: str, path: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "show", f"{ref}:{path}"],
            cwd=root,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise OperationalTwinProtocolError(
            f"successor_bound_git_input_missing: {ref}:{path}"
        ) from error


def _git_json(root: Path, ref: str, path: str) -> dict[str, object]:
    try:
        value = json.loads(_git_bytes(root, ref, path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OperationalTwinProtocolError(
            f"successor_bound_git_json_invalid: {ref}:{path}"
        ) from error
    if not isinstance(value, dict):
        raise OperationalTwinProtocolError(
            f"successor_bound_git_json_invalid: {ref}:{path}"
        )
    return value


def _git_commit(root: Path, value: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--verify", f"{value}^{{commit}}"],
            cwd=root,
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise OperationalTwinProtocolError(
            f"successor_bound_commit_missing: {value}"
        ) from error


def _current_commit(root: Path) -> str:
    return _git_commit(root, "HEAD")


def _is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=root,
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0


def _file_identity(root: Path, path: str) -> dict[str, object]:
    content = read_repo_bytes(root, path)
    git_object_id = _git_object(root, "HEAD", path)
    if git_object_id is not None and _git_bytes(root, "HEAD", path) != content:
        git_object_id = None
    return {
        "path": path,
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
        "git_object_id": git_object_id,
    }


def _git_identity(root: Path, ref: str, path: str) -> dict[str, object]:
    content = _git_bytes(root, ref, path)
    object_id = _git_object(root, ref, path)
    if object_id is None:
        raise OperationalTwinProtocolError(
            f"successor_bound_git_input_missing: {ref}:{path}"
        )
    return {
        "path": path,
        "git_object_id": object_id,
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }


def _identity_basis(identity: Mapping[str, object]) -> dict[str, object]:
    return {
        "path": identity["path"],
        "sha256": identity["sha256"],
        "size_bytes": identity["size_bytes"],
    }


def _uid(prefix: str, basis: object) -> str:
    return f"{prefix}_{canonical_hash(basis)[:24]}"


def _without_uid(value: Mapping[str, object], field: str) -> dict[str, object]:
    result = deepcopy(dict(value))
    result.pop(field, None)
    return result


def compute_environment_contract_uid(config: Mapping[str, object]) -> str:
    contract = _contract(config)
    policy = dict(contract["identity_policy"])
    return _uid(
        str(policy["environment_contract_uid_prefix"]),
        _without_uid(dict(contract["environment_contract"]), "environment_contract_uid"),
    )


def compute_resource_contract_uid(config: Mapping[str, object]) -> str:
    contract = _contract(config)
    policy = dict(contract["identity_policy"])
    return _uid(
        str(policy["resource_contract_uid_prefix"]),
        _without_uid(dict(contract["resource_contract"]), "resource_contract_uid"),
    )


def successor_config_identity_basis(config: Mapping[str, object]) -> dict[str, object]:
    contract = deepcopy(_contract(config))
    contract.pop("identity", None)
    contract["environment_contract"] = _without_uid(
        dict(contract["environment_contract"]), "environment_contract_uid"
    )
    contract["resource_contract"] = _without_uid(
        dict(contract["resource_contract"]), "resource_contract_uid"
    )
    return contract


def compute_config_contract_uid(config: Mapping[str, object]) -> str:
    contract = _contract(config)
    policy = dict(contract["identity_policy"])
    return _uid(
        str(policy["config_contract_uid_prefix"]),
        successor_config_identity_basis(config),
    )


def compute_schema_contract_uid(config: Mapping[str, object], root: Path) -> str:
    contract = _contract(config)
    policy = dict(contract["identity_policy"])
    path = "Libs/configs/schema/operational_twin_successor_lineage_protocol.schema.yaml"
    content = read_repo_bytes(root, path)
    return _uid(
        str(policy["schema_contract_uid_prefix"]),
        {"path": path, "sha256": hashlib.sha256(content).hexdigest()},
    )


def successor_source_identities(
    config: Mapping[str, object], root: Path
) -> list[dict[str, object]]:
    return [_file_identity(root, str(path)) for path in _contract(config)["source_bindings"]]


def compute_source_contract_uid(config: Mapping[str, object], root: Path) -> str:
    contract = _contract(config)
    policy = dict(contract["identity_policy"])
    basis = [_identity_basis(row) for row in successor_source_identities(config, root)]
    return _uid(str(policy["source_contract_uid_prefix"]), basis)


def _inherited_protocol(config: Mapping[str, object]) -> dict[str, object]:
    key = str(dict(_contract(config)["invariant_contract"])["inherited_protocol_config_key"])
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise OperationalTwinProtocolError(
            f"successor_inherited_protocol_missing: {key}"
        )
    return deepcopy(dict(value))


def _lineage_identity_basis(
    config: Mapping[str, object], identities: Mapping[str, str]
) -> dict[str, object]:
    contract = _contract(config)
    predecessor = dict(contract["predecessor_authority"])
    return {
        "authoritative_date": contract["authoritative_date"],
        "evidence_class": contract["evidence_class"],
        "predecessor_candidate_lineage_uid": predecessor[
            "predecessor_candidate_lineage_uid"
        ],
        "predecessor_relationship": dict(contract["supersession_contract"])[
            "relationship"
        ],
        "round_48_decision_uid": dict(predecessor["round_48"])["decision_uid"],
        "round_48_terminal": dict(predecessor["round_48"])["terminal_decision"],
        "initialization": contract["initialization"],
        "invariant_contract": contract["invariant_contract"],
        "output_namespace": contract["output_namespace"],
        "config_contract_uid": identities["config_contract_uid"],
        "schema_contract_uid": identities["schema_contract_uid"],
        "source_contract_uid": identities["source_contract_uid"],
        "environment_contract_uid": identities["environment_contract_uid"],
        "resource_contract_uid": identities["resource_contract_uid"],
    }


def _derived_output_paths(
    contract: Mapping[str, object], candidate_lineage_uid: str
) -> dict[str, object]:
    namespace = dict(contract["output_namespace"])
    return {
        "protocol_root": namespace["protocol_root"],
        "candidate_lineage_manifest": namespace["candidate_lineage_manifest"],
        "assignment_manifest": namespace["assignment_manifest"],
        "assignment_commands": namespace["assignment_commands"],
        "preflight_authority": namespace["preflight_authority"],
        "preflight_report": namespace["preflight_report"],
        "attempt_root_template": namespace["attempt_root_template"],
        "candidate_calibration_execution_authority": str(
            namespace["calibration_execution_authority_template"]
        ).format(candidate_lineage_uid=candidate_lineage_uid),
    }


def _derived_protocol_basis(
    config: Mapping[str, object], identities: Mapping[str, str]
) -> dict[str, object]:
    contract = _contract(config)
    inherited = _inherited_protocol(config)
    candidate_lineage_uid = identities["candidate_lineage_uid"]
    execution_authority = deepcopy(dict(inherited["execution_authority_contract"]))
    review = dict(contract["review_contract"])
    namespace = dict(contract["output_namespace"])
    execution_authority.update(
        {
            "approved_protocol_review_path": review["path"],
            "required_review_schema_version": review["schema_version"],
            "required_review_round_id": review["round_id"],
            "required_review_direction_id": review["direction_id"],
            "required_review_decision": review["decision"],
            "candidate_execution_authority_path_template": namespace[
                "execution_authority_template"
            ],
            "resource_environment_receipt_path_template": namespace[
                "resource_receipt_template"
            ],
            "authority_bound_in_this_round": False,
            "resource_receipt_bound_in_this_round": False,
        }
    )
    return {
        "schema_version": contract["schema_version"],
        "authoritative_date": contract["authoritative_date"],
        "evidence_class": contract["evidence_class"],
        "candidate_lineage_uid": candidate_lineage_uid,
        "parent_candidate_lineage_uid": dict(contract["predecessor_authority"])[
            "predecessor_candidate_lineage_uid"
        ],
        "parent_relationship": dict(contract["supersession_contract"])[
            "relationship"
        ],
        "initialization": contract["initialization"],
        "identity_policy": contract["identity_policy"],
        "scientific_contract": inherited["scientific_contract"],
        "historical_boundary": inherited["historical_boundary"],
        "sealed_manifest_metadata": inherited["sealed_manifest_metadata"],
        "scheduler_evidence_contract": inherited["scheduler_evidence_contract"],
        "supersession_contract": contract["supersession_contract"],
        "predecessor_authority": contract["predecessor_authority"],
        "execution_authority_contract": execution_authority,
        "execution_contract": inherited["execution_contract"],
        "output_paths": _derived_output_paths(contract, candidate_lineage_uid),
        "review_contract": contract["review_contract"],
        "identity_contracts": {
            key: identities[key]
            for key in (
                "config_contract_uid",
                "schema_contract_uid",
                "source_contract_uid",
                "environment_contract_uid",
                "resource_contract_uid",
            )
        },
        "source_bindings": contract["source_bindings"],
        "access_counters": contract["access_counters"],
        "closed_flags": contract["closed_flags"],
        "terminal_policy": contract["terminal_policy"],
    }


def compute_successor_identity_declarations(
    config: Mapping[str, object],
    root: Path,
    pre_change_manifest: Mapping[str, object] | None = None,
) -> dict[str, str | None]:
    contract = _contract(config)
    policy = dict(contract["identity_policy"])
    identities: dict[str, str | None] = {
        "config_contract_uid": compute_config_contract_uid(config),
        "schema_contract_uid": compute_schema_contract_uid(config, root),
        "source_contract_uid": compute_source_contract_uid(config, root),
        "environment_contract_uid": compute_environment_contract_uid(config),
        "resource_contract_uid": compute_resource_contract_uid(config),
    }
    lineage_basis = _lineage_identity_basis(
        config, {key: str(value) for key, value in identities.items()}
    )
    identities["candidate_lineage_uid"] = _uid(
        str(policy["candidate_lineage_uid_prefix"]), lineage_basis
    )
    protocol_basis = _derived_protocol_basis(
        config, {key: str(value) for key, value in identities.items()}
    )
    identities["protocol_uid"] = _uid(
        str(policy["protocol_uid_prefix"]), protocol_basis
    )
    identities["preflight_uid"] = None
    if pre_change_manifest is not None:
        protocol = derive_successor_protocol(
            config,
            root,
            identities={key: str(value) for key, value in identities.items() if value},
        )
        assignments = build_candidate_assignments(protocol)
        mappings = build_predecessor_bijections(config, root, protocol, assignments)
        structural = _structural_identities(protocol, assignments, mappings)
        pre_change_hash = str(pre_change_manifest.get("manifest_hash", ""))
        preflight_basis = {
            "authoritative_date": contract["authoritative_date"],
            "candidate_lineage_uid": identities["candidate_lineage_uid"],
            "protocol_uid": identities["protocol_uid"],
            **{
                key: identities[key]
                for key in (
                    "config_contract_uid",
                    "schema_contract_uid",
                    "source_contract_uid",
                    "environment_contract_uid",
                    "resource_contract_uid",
                )
            },
            **structural,
            "pre_change_identity_manifest_hash": pre_change_hash,
        }
        identities["preflight_uid"] = _uid(
            str(policy["preflight_uid_prefix"]), preflight_basis
        )
    return identities


def derive_successor_protocol(
    config: Mapping[str, object],
    root: Path,
    *,
    identities: Mapping[str, str] | None = None,
) -> dict[str, object]:
    expected = dict(identities or compute_successor_identity_declarations(config, root))
    if not expected.get("candidate_lineage_uid") or not expected.get("protocol_uid"):
        raise OperationalTwinProtocolError("successor_identity_derivation_incomplete")
    basis = _derived_protocol_basis(config, expected)
    basis["protocol_uid"] = expected["protocol_uid"]
    return basis


def _predecessor_protocol_artifacts(
    config: Mapping[str, object], root: Path, ref: str
) -> list[dict[str, object]]:
    inherited = _inherited_protocol(config)
    amendment = dict(dict(inherited["supersession_contract"])["resource_receipt_amendment_basis"])
    generations = deepcopy(dict(amendment["predecessor_generations"]))
    current_paths = dict(inherited["output_paths"])
    generations["execution_interface_guarded"] = {
        "protocol_uid": inherited["protocol_uid"],
        "artifacts": {
            "candidate_lineage_manifest": {"path": current_paths["candidate_lineage_manifest"]},
            "assignment_manifest": {"path": current_paths["assignment_manifest"]},
            "assignment_commands": {"path": current_paths["assignment_commands"]},
            "preflight_authority": {"path": current_paths["preflight_authority"]},
            "preflight_report": {"path": current_paths["preflight_report"]},
        },
    }
    result: list[dict[str, object]] = []
    for generation_name, generation_value in generations.items():
        generation = dict(generation_value)
        for role, identity_value in dict(generation["artifacts"]).items():
            identity = _git_identity(root, ref, str(dict(identity_value)["path"]))
            configured_sha = dict(identity_value).get("sha256")
            if configured_sha is not None and identity["sha256"] != configured_sha:
                raise OperationalTwinProtocolError(
                    f"successor_predecessor_sha_mismatch: {generation_name}.{role}"
                )
            result.append(
                {
                    "generation": generation_name,
                    "protocol_uid": generation["protocol_uid"],
                    "role": role,
                    **identity,
                }
            )
    return result


def publication_paths(config: Mapping[str, object]) -> dict[str, str]:
    namespace = dict(_contract(config)["output_namespace"])
    return {key: str(namespace[key]) for key in PUBLICATION_KEYS}


def _future_paths(
    config: Mapping[str, object], protocol: Mapping[str, object], assignments: Sequence[Mapping[str, object]]
) -> list[dict[str, str]]:
    namespace = dict(_contract(config)["output_namespace"])
    lineage = str(protocol["candidate_lineage_uid"])
    paths: list[dict[str, str]] = []
    for role in (
        "candidate_root_template",
        "calibration_execution_authority_template",
        "execution_root_template",
        "dispatch_manifest_template",
        "recovery_eligibility_template",
        "scheduler_receipts_root_template",
        "logs_root_template",
        "aggregation_root_template",
        "candidate_execution_catalog_template",
    ):
        paths.append(
            {
                "role": role,
                "path": str(namespace[role]).format(candidate_lineage_uid=lineage),
            }
        )
    for assignment_value in assignments:
        assignment = dict(assignment_value)
        for attempt_value in assignment["attempts"]:
            attempt = dict(attempt_value)
            paths.extend(
                [
                    {"role": "attempt_root", "path": str(attempt["output_root"])},
                    {
                        "role": "execution_authority",
                        "path": str(attempt["execution_authority"]),
                    },
                    {
                        "role": "resource_environment_receipt",
                        "path": str(attempt["resource_environment_receipt"]),
                    },
                ]
            )
    unique: dict[str, str] = {}
    for row in paths:
        unique.setdefault(row["path"], row["role"])
    return [{"role": role, "path": path} for path, role in unique.items()]


def build_pre_change_identity_manifest(
    config: Mapping[str, object], root: Path
) -> dict[str, object]:
    contract = _contract(config)
    predecessor = dict(contract["predecessor_authority"])
    planner_commit = str(predecessor["planner_commit"])
    identities = compute_successor_identity_declarations(config, root)
    protocol = derive_successor_protocol(
        config,
        root,
        identities={key: str(value) for key, value in identities.items() if value},
    )
    assignments = build_candidate_assignments(protocol)
    historical = _predecessor_protocol_artifacts(config, root, planner_commit)
    authority_paths = []
    for section_name, names in (
        ("round_47", ("review_record", "closeout_record", "audit", "report")),
        ("round_48", ("review_record", "closeout_record", "decision_audit", "decision_report")),
    ):
        section = dict(predecessor[section_name])
        for name in names:
            binding = dict(section[name])
            identity = _git_identity(root, planner_commit, str(binding["path"]))
            if identity["git_object_id"] != binding["git_object_id"]:
                raise OperationalTwinProtocolError(
                    f"successor_{section_name}_{name}_identity_mismatch"
                )
            authority_paths.append({"authority": f"{section_name}.{name}", **identity})
    for authority, binding_value in (
        ("round_43.review_record", predecessor["round_43_review"]),
        ("round_34.audit", dict(predecessor["round_34"])["audit"]),
    ):
        binding = dict(binding_value)
        identity = _git_identity(root, planner_commit, str(binding["path"]))
        if identity["git_object_id"] != binding["git_object_id"]:
            raise OperationalTwinProtocolError(
                f"successor_{authority.replace('.', '_')}_identity_mismatch"
            )
        authority_paths.append({"authority": authority, **identity})
    for role, binding_value in dict(dict(predecessor["round_34"])["manifests"]).items():
        binding = dict(binding_value)
        identity = _git_identity(root, planner_commit, str(binding["path"]))
        if identity["sha256"] != binding["sha256"]:
            raise OperationalTwinProtocolError(
                f"successor_round_34_{role}_sha_mismatch"
            )
        authority_paths.append({"authority": f"round_34.{role}", **identity})
    absent = []
    protocol_root = str(dict(contract["output_namespace"])["protocol_root"])
    absent.append(
        {
            "role": "protocol_root",
            "path": protocol_root,
            "git_object_id_at_planner_commit": _git_object(
                root, planner_commit, protocol_root
            ),
            "filesystem_absent_when_recorded": not os.path.lexists(
                root / protocol_root
            ),
        }
    )
    for role, path in publication_paths(config).items():
        absent.append(
            {
                "role": role,
                "path": path,
                "git_object_id_at_planner_commit": _git_object(
                    root, planner_commit, path
                ),
                "filesystem_absent_when_recorded": not os.path.lexists(
                    root / path
                ),
            }
        )
    for row in _future_paths(config, protocol, assignments):
        absent.append(
            {
                **row,
                "git_object_id_at_planner_commit": _git_object(
                    root, planner_commit, row["path"]
                ),
                "filesystem_absent_when_recorded": not os.path.lexists(
                    root / row["path"]
                ),
            }
        )
    retry_paths = dict(config["operational_twin_candidate_retry_authority"])[
        "output_paths"
    ]
    reserved_retry_paths = []
    for key in (
        "candidate_lineage_manifest",
        "assignment_manifest",
        "assignment_commands",
        "preflight_authority",
        "preflight_report",
    ):
        path = str(dict(retry_paths)[key])
        reserved_retry_paths.append(
            {
                "role": key,
                "path": path,
                "git_object_id_at_planner_commit": _git_object(
                    root, planner_commit, path
                ),
                "filesystem_absent_when_recorded": not os.path.lexists(
                    root / path
                ),
            }
        )
    round_46 = dict(predecessor["round_46"])
    for role, path_key, identity_key in (
        ("candidate", "candidate_tree_path", "candidate_tree_id"),
        ("audit", "audit_tree_path", "audit_tree_id"),
    ):
        observed = _git_object(
            root, str(round_46["merged_commit"]), str(round_46[path_key])
        )
        if observed != round_46[identity_key]:
            raise OperationalTwinProtocolError(
                f"successor_round_46_{role}_tree_identity_mismatch"
            )
    payload: dict[str, object] = {
        "schema_version": PRE_CHANGE_SCHEMA_VERSION,
        "authoritative_date": contract["authoritative_date"],
        "recorded_against_commit": planner_commit,
        "baseline_commit": predecessor["baseline_commit"],
        "candidate_lineage_uid": identities["candidate_lineage_uid"],
        "protocol_uid": identities["protocol_uid"],
        "historical_protocol_artifacts": historical,
        "authority_artifacts": authority_paths,
        "historical_trees": [
            {
                "role": "round_46_candidate_tree",
                "commit": round_46["merged_commit"],
                "path": round_46["candidate_tree_path"],
                "git_object_id": round_46["candidate_tree_id"],
            },
            {
                "role": "round_46_audit_tree",
                "commit": round_46["merged_commit"],
                "path": round_46["audit_tree_path"],
                "git_object_id": round_46["audit_tree_id"],
            },
        ],
        "successor_paths_absent": absent,
        "reserved_retry_paths_absent": reserved_retry_paths,
    }
    payload["manifest_hash"] = canonical_hash(payload)
    return payload


def validate_pre_change_identity_manifest(
    config: Mapping[str, object], root: Path, manifest: Mapping[str, object]
) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    expected = build_pre_change_identity_manifest(config, root)

    declared = deepcopy(dict(manifest))
    declared_hash = declared.pop("manifest_hash", None)
    if declared_hash != canonical_hash(declared):
        blockers.append(
            _block(
                "successor_pre_change_identity_manifest_hash_mismatch",
                "The pre-change identity manifest hash does not recompute.",
            )
        )

    expected_without_hash = deepcopy(expected)
    expected_without_hash.pop("manifest_hash", None)
    for payload in (declared, expected_without_hash):
        for key in ("successor_paths_absent", "reserved_retry_paths_absent"):
            for row in payload.get(key, []):
                row["filesystem_absent_when_recorded"] = True
    if declared != expected_without_hash:
        blockers.append(
            _block(
                "successor_pre_change_identity_manifest_drift",
                "Pre-change identities or required absence records do not recompute.",
            )
        )
    if any(
        row.get("git_object_id_at_planner_commit") is not None
        or row.get("filesystem_absent_when_recorded") is not True
        for row in manifest.get("successor_paths_absent", [])
    ):
        blockers.append(
            _block(
                "successor_pre_change_path_not_absent",
                "Every successor publication and future execution path must be absent before publication.",
            )
        )
    if any(
        row.get("git_object_id_at_planner_commit") is not None
        or row.get("filesystem_absent_when_recorded") is not True
        for row in manifest.get("reserved_retry_paths_absent", [])
    ):
        blockers.append(
            _block(
                "successor_reserved_retry_path_not_absent",
                "The five unissued existing-lineage retry artifacts must remain absent.",
            )
        )
    return blockers


def build_predecessor_bijections(
    config: Mapping[str, object],
    root: Path,
    protocol: Mapping[str, object],
    assignments: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    inherited = _inherited_protocol(config)
    predecessor_path = str(dict(inherited["output_paths"])["assignment_manifest"])
    predecessor = read_repo_json(root, predecessor_path)
    predecessor_rows = {
        (str(row["method_id"]), int(row["seed"])): dict(row)
        for row in predecessor["assignments"]
    }
    assignment_rows: list[dict[str, object]] = []
    attempt_rows: list[dict[str, object]] = []
    for successor_value in assignments:
        successor = dict(successor_value)
        key = (str(successor["method_id"]), int(successor["seed"]))
        old = predecessor_rows.get(key)
        if old is None:
            raise OperationalTwinProtocolError(
                f"successor_predecessor_assignment_missing: {key}"
            )
        old_attempts = {
            str(row["attempt_kind"]): dict(row) for row in old["attempts"]
        }
        new_attempts = {
            str(row["attempt_kind"]): dict(row) for row in successor["attempts"]
        }
        assignment_rows.append(
            {
                "method_id": key[0],
                "seed": key[1],
                "predecessor_candidate_lineage_uid": old["candidate_lineage_uid"],
                "predecessor_protocol_uid": old["protocol_uid"],
                "predecessor_assignment_uid": old["assignment_uid"],
                "successor_candidate_lineage_uid": successor[
                    "candidate_lineage_uid"
                ],
                "successor_protocol_uid": successor["protocol_uid"],
                "successor_assignment_uid": successor["assignment_uid"],
            }
        )
        for attempt_kind in EXPECTED_ATTEMPT_KINDS:
            old_attempt = old_attempts[attempt_kind]
            new_attempt = new_attempts[attempt_kind]
            attempt_rows.append(
                {
                    "method_id": key[0],
                    "seed": key[1],
                    "attempt_kind": attempt_kind,
                    "attempt_ordinal": new_attempt["attempt_ordinal"],
                    "conditional": new_attempt["conditional"],
                    "eligible_after_failure_class": new_attempt[
                        "eligible_after_failure_class"
                    ],
                    "predecessor_assignment_uid": old["assignment_uid"],
                    "successor_assignment_uid": successor["assignment_uid"],
                    "predecessor_attempt_uid": old_attempt["attempt_uid"],
                    "successor_attempt_uid": new_attempt["attempt_uid"],
                    "predecessor_output_root": old_attempt["output_root"],
                    "successor_output_root": new_attempt["output_root"],
                    "predecessor_execution_authority": old_attempt[
                        "execution_authority"
                    ],
                    "successor_execution_authority": new_attempt[
                        "execution_authority"
                    ],
                    "predecessor_resource_environment_receipt": old_attempt[
                        "resource_environment_receipt"
                    ],
                    "successor_resource_environment_receipt": new_attempt[
                        "resource_environment_receipt"
                    ],
                }
            )
    return {
        "assignment_bijection": assignment_rows,
        "assignment_bijection_hash": canonical_hash(assignment_rows),
        "attempt_slot_bijection": attempt_rows,
        "attempt_slot_bijection_hash": canonical_hash(attempt_rows),
    }


def _structural_identities(
    protocol: Mapping[str, object],
    assignments: Sequence[Mapping[str, object]],
    mappings: Mapping[str, object],
) -> dict[str, str]:
    assignment_identity = [
        {
            "candidate_lineage_uid": row["candidate_lineage_uid"],
            "protocol_uid": row["protocol_uid"],
            "assignment_uid": row["assignment_uid"],
            "method_id": row["method_id"],
            "seed": row["seed"],
        }
        for row in assignments
    ]
    attempt_identity = [
        {
            "assignment_uid": row["assignment_uid"],
            "attempt_uid": attempt["attempt_uid"],
            "attempt_kind": attempt["attempt_kind"],
            "attempt_ordinal": attempt["attempt_ordinal"],
            "conditional": attempt["conditional"],
            "eligible_after_failure_class": attempt[
                "eligible_after_failure_class"
            ],
            "output_root": attempt["output_root"],
            "execution_authority": attempt["execution_authority"],
            "resource_environment_receipt": attempt[
                "resource_environment_receipt"
            ],
        }
        for row in assignments
        for attempt in row["attempts"]
    ]
    command_identity = []
    for attempt_kind in EXPECTED_ATTEMPT_KINDS:
        for row in assignments:
            attempt = next(
                item
                for item in row["attempts"]
                if item["attempt_kind"] == attempt_kind
            )
            command_identity.append(
                {
                    "command_index": len(command_identity),
                    "assignment_uid": row["assignment_uid"],
                    "attempt_uid": attempt["attempt_uid"],
                    "method_id": row["method_id"],
                    "seed": row["seed"],
                    "resource_profile": row["resource_profile"],
                    "resource_requirements": row["resource_requirements"],
                    "wall_time": row["wall_time"],
                    "output_root": attempt["output_root"],
                    "execution_authority": attempt["execution_authority"],
                    "resource_environment_receipt": attempt[
                        "resource_environment_receipt"
                    ],
                }
            )
    return {
        "assignment_identity_hash": canonical_hash(assignment_identity),
        "attempt_identity_hash": canonical_hash(attempt_identity),
        "command_identity_hash": canonical_hash(command_identity),
        "assignment_bijection_hash": str(mappings["assignment_bijection_hash"]),
        "attempt_slot_bijection_hash": str(mappings["attempt_slot_bijection_hash"]),
        "scientific_contract_hash": canonical_hash(protocol["scientific_contract"]),
        "execution_contract_hash": canonical_hash(protocol["execution_contract"]),
        "scheduler_evidence_contract_hash": canonical_hash(
            protocol["scheduler_evidence_contract"]
        ),
    }


def _successor_initialization_receipt(
    contract: Mapping[str, object], protocol: Mapping[str, object]
) -> dict[str, object]:
    initialization = dict(contract["initialization"])
    return {
        "initialization_mode": initialization["initialization_mode"],
        "provenance_parent_candidate_lineage_uid": protocol[
            "parent_candidate_lineage_uid"
        ],
        "initialization_source_candidate_lineage_uid": None,
        "resume_allowed": initialization["resume_allowed"],
        **{
            field: initialization[field]
            for field in CANDIDATE_FORBIDDEN_STATE_INPUT_FIELDS
        },
        "historical_state_use_allowed": initialization[
            "historical_state_use_allowed"
        ],
    }


def build_successor_artifacts(
    config: Mapping[str, object], root: Path, pre_change_manifest: Mapping[str, object]
) -> dict[str, object]:
    contract = _contract(config)
    identities = compute_successor_identity_declarations(
        config, root, pre_change_manifest
    )
    protocol = derive_successor_protocol(
        config,
        root,
        identities={key: str(value) for key, value in identities.items() if value},
    )
    protocol["preflight_uid"] = identities["preflight_uid"]
    assignments = build_candidate_assignments(protocol)
    mappings = build_predecessor_bijections(config, root, protocol, assignments)
    structural = _structural_identities(protocol, assignments, mappings)
    initialization = _successor_initialization_receipt(contract, protocol)
    candidate: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "FROZEN",
        "terminal_decision": POSITIVE_TERMINAL,
        "authoritative_date": contract["authoritative_date"],
        "evidence_class": contract["evidence_class"],
        "candidate_lineage_uid": identities["candidate_lineage_uid"],
        "protocol_uid": identities["protocol_uid"],
        "preflight_uid": identities["preflight_uid"],
        "predecessor_candidate_lineage_uid": dict(contract["predecessor_authority"])[
            "predecessor_candidate_lineage_uid"
        ],
        "predecessor_relationship": dict(contract["supersession_contract"])[
            "relationship"
        ],
        "initialization": initialization,
        "scientific_contract_hash": structural["scientific_contract_hash"],
        "execution_contract_hash": structural["execution_contract_hash"],
        "scheduler_evidence_contract_hash": structural[
            "scheduler_evidence_contract_hash"
        ],
        "identity_contracts": protocol["identity_contracts"],
        "assignment_uids": [row["assignment_uid"] for row in assignments],
        "attempt_uids": [
            attempt["attempt_uid"]
            for row in assignments
            for attempt in row["attempts"]
        ],
        "assignment_bijection": mappings["assignment_bijection"],
        "assignment_bijection_hash": mappings["assignment_bijection_hash"],
        "output_namespace": contract["output_namespace"],
        "review_contract": contract["review_contract"],
        "candidate_dispatch_authorized": False,
        "execution_handoff_required": True,
        "scientific_twin_result": "NOT_RUN",
        "access_counters": contract["access_counters"],
        "closed_flags": contract["closed_flags"],
    }
    candidate["candidate_manifest_hash"] = canonical_hash(candidate)
    assignment_manifest: dict[str, object] = {
        "schema_version": ASSIGNMENT_SCHEMA_VERSION,
        "candidate_lineage_uid": identities["candidate_lineage_uid"],
        "protocol_uid": identities["protocol_uid"],
        "preflight_uid": identities["preflight_uid"],
        "candidate_manifest_hash": candidate["candidate_manifest_hash"],
        "assignment_count": len(assignments),
        "attempt_slot_count": sum(len(row["attempts"]) for row in assignments),
        "method_ids": REQUIRED_METHOD_IDS,
        "seeds": EXPECTED_SEEDS,
        "assignments": assignments,
        "attempt_slot_bijection": mappings["attempt_slot_bijection"],
        "attempt_slot_bijection_hash": mappings["attempt_slot_bijection_hash"],
        "initialization": initialization,
        "environment_contract_uid": identities["environment_contract_uid"],
        "resource_contract_uid": identities["resource_contract_uid"],
        "candidate_dispatch_authorized": False,
        "access_counters": contract["access_counters"],
    }
    assignment_manifest["assignment_manifest_hash"] = canonical_hash(
        assignment_manifest
    )
    runtime_config = deepcopy(dict(config))
    runtime_config["operational_twin_candidate_lineage_protocol"] = protocol
    commands = build_candidate_commands(
        runtime_config, candidate, assignment_manifest
    )
    commands["schema_version"] = COMMAND_SCHEMA_VERSION
    commands["assignment_bijection_hash"] = mappings["assignment_bijection_hash"]
    commands["attempt_slot_bijection_hash"] = mappings[
        "attempt_slot_bijection_hash"
    ]
    commands["execution_handoff_required"] = True
    commands["candidate_dispatch_authorized"] = False
    commands["command_catalog_hash"] = command_catalog_hash(commands)
    for row in commands["commands"]:
        argv = list(row["argv"])
        argv[argv.index("--command-catalog-hash") + 1] = commands[
            "command_catalog_hash"
        ]
        row["argv"] = argv
    return {
        "protocol": protocol,
        "identities": identities,
        "structural_identities": structural,
        "mappings": mappings,
        "candidate_lineage_manifest": candidate,
        "assignment_manifest": assignment_manifest,
        "assignment_commands": commands,
    }


def _read_pre_change_manifest(
    config: Mapping[str, object], root: Path, path: Path | None = None
) -> tuple[dict[str, object], dict[str, object]]:
    configured = str(dict(_contract(config)["evidence_inputs"])["pre_change_identity_manifest"])
    supplied = path or (root / configured)
    supplied_resolved = supplied.resolve()
    expected_resolved = (root / configured).resolve()
    if supplied_resolved != expected_resolved:
        raise OperationalTwinProtocolError(
            "successor_pre_change_identity_manifest_path_mismatch"
        )
    try:
        payload = json.loads(supplied_resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OperationalTwinProtocolError(
            "successor_pre_change_identity_manifest_unavailable"
        ) from error
    if not isinstance(payload, dict):
        raise OperationalTwinProtocolError(
            "successor_pre_change_identity_manifest_invalid"
        )
    return payload, _file_identity(root, configured)


def _read_validation_evidence(
    config: Mapping[str, object], root: Path, path: Path | None = None
) -> tuple[dict[str, object], dict[str, object]]:
    configured = str(dict(_contract(config)["evidence_inputs"])["validation_evidence"])
    supplied = path or (root / configured)
    if supplied.resolve() != (root / configured).resolve():
        raise OperationalTwinProtocolError(
            "successor_validation_evidence_path_mismatch"
        )
    try:
        payload = json.loads(supplied.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OperationalTwinProtocolError(
            "successor_validation_evidence_unavailable"
        ) from error
    if not isinstance(payload, dict):
        raise OperationalTwinProtocolError("successor_validation_evidence_invalid")
    return payload, _file_identity(root, configured)


def _validate_path(
    root: Path,
    value: str,
    *,
    allowed_root: str,
    require_absent: bool,
    require_git_absent: bool,
) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    declared = Path(value)
    if declared.is_absolute() or ".." in declared.parts or value != declared.as_posix():
        return [
            _block(
                "successor_path_not_canonical",
                f"{value} must be one repository-relative canonical POSIX path.",
            )
        ]
    try:
        declared.relative_to(Path(allowed_root))
    except ValueError:
        return [
            _block(
                "successor_path_outside_namespace",
                f"{value} is outside {allowed_root}.",
            )
        ]
    current = root.resolve()
    for component in declared.parts:
        current = current / component
        if not os.path.lexists(current):
            continue
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            blockers.append(
                _block("successor_path_symlink", f"{value} contains a symlink.")
            )
            break
    if require_absent and os.path.lexists(root / declared):
        blockers.append(
            _block("successor_path_collision", f"{value} already exists.")
        )
    if require_git_absent and _git_object(root, "HEAD", value) is not None:
        blockers.append(
            _block("successor_git_path_collision", f"{value} is tracked already.")
        )
    return blockers


def _validate_bijections(
    built: Mapping[str, object], predecessor_lineage_uid: str
) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    mappings = dict(built["mappings"])
    assignment_rows = list(mappings["assignment_bijection"])
    attempt_rows = list(mappings["attempt_slot_bijection"])
    if len(assignment_rows) != 15:
        blockers.append(_block("successor_assignment_bijection_incomplete", "Expected 15 rows."))
    if len(attempt_rows) != 30:
        blockers.append(_block("successor_attempt_bijection_incomplete", "Expected 30 rows."))
    expected_keys = [
        (method, seed)
        for method in REQUIRED_METHOD_IDS
        for seed in EXPECTED_SEEDS
    ]
    observed_assignment_keys = [
        (str(row["method_id"]), int(row["seed"])) for row in assignment_rows
    ]
    if observed_assignment_keys != expected_keys:
        blockers.append(_block("successor_assignment_bijection_order_drift", "Method-seed order drifted."))
    observed_attempt_keys = [
        (str(row["method_id"]), int(row["seed"]), str(row["attempt_kind"]))
        for row in attempt_rows
    ]
    expected_attempt_keys = [
        (method, seed, kind)
        for method in REQUIRED_METHOD_IDS
        for seed in EXPECTED_SEEDS
        for kind in EXPECTED_ATTEMPT_KINDS
    ]
    if observed_attempt_keys != expected_attempt_keys:
        blockers.append(_block("successor_attempt_bijection_order_drift", "Attempt mapping order drifted."))
    for field in (
        "predecessor_assignment_uid",
        "successor_assignment_uid",
    ):
        values = [str(row[field]) for row in assignment_rows]
        if len(values) != len(set(values)):
            blockers.append(_block("successor_assignment_bijection_not_one_to_one", field))
    for field in (
        "predecessor_attempt_uid",
        "successor_attempt_uid",
        "predecessor_output_root",
        "successor_output_root",
        "predecessor_execution_authority",
        "successor_execution_authority",
        "predecessor_resource_environment_receipt",
        "successor_resource_environment_receipt",
    ):
        values = [str(row[field]) for row in attempt_rows]
        if len(values) != len(set(values)):
            blockers.append(_block("successor_attempt_bijection_not_one_to_one", field))
    if any(
        row["predecessor_assignment_uid"] == row["successor_assignment_uid"]
        or row["predecessor_candidate_lineage_uid"]
        != predecessor_lineage_uid
        or row["successor_candidate_lineage_uid"] == predecessor_lineage_uid
        for row in assignment_rows
    ):
        blockers.append(_block("successor_assignment_identity_collision", "Old and new assignment identities overlap."))
    if any(
        row["predecessor_attempt_uid"] == row["successor_attempt_uid"]
        or row["predecessor_output_root"] == row["successor_output_root"]
        or row["predecessor_execution_authority"]
        == row["successor_execution_authority"]
        or row["predecessor_resource_environment_receipt"]
        == row["successor_resource_environment_receipt"]
        for row in attempt_rows
    ):
        blockers.append(_block("successor_attempt_identity_collision", "Old and new attempt identities or paths overlap."))
    if mappings["assignment_bijection_hash"] != canonical_hash(assignment_rows):
        blockers.append(_block("successor_assignment_bijection_hash_mismatch", "Assignment mapping hash drifted."))
    if mappings["attempt_slot_bijection_hash"] != canonical_hash(attempt_rows):
        blockers.append(_block("successor_attempt_bijection_hash_mismatch", "Attempt mapping hash drifted."))
    return blockers


def validate_successor_config(
    config: Mapping[str, object],
    root: Path,
    *,
    pre_change_manifest: Mapping[str, object] | None = None,
    require_committed_sources: bool = False,
) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    try:
        contract = _contract(config)
        schema = _schema(config)
    except OperationalTwinProtocolError as error:
        return [_block("successor_config_missing", str(error))]
    required_top = schema.get("required_top_level_fields", [])
    if set(contract) != set(required_top):
        blockers.append(_block("successor_config_top_level_fields_mismatch", "Successor config fields drifted."))
    if contract.get("schema_version") != schema.get("protocol_schema_version"):
        blockers.append(_block("successor_schema_version_mismatch", "Protocol/schema versions differ."))
    if contract.get("authoritative_date") != schema.get("required_authoritative_date"):
        blockers.append(_block("successor_authoritative_date_mismatch", "Authoritative date drifted."))
    if contract.get("evidence_class") != schema.get("required_evidence_class"):
        blockers.append(_block("successor_evidence_class_mismatch", "Evidence class drifted."))
    identity = dict(contract.get("identity", {}))
    if set(identity) != set(schema.get("required_identity_fields", [])):
        blockers.append(_block("successor_identity_fields_mismatch", "Identity declarations drifted."))
    initialization = dict(contract.get("initialization", {}))
    if initialization.get("initialization_mode") != "fresh_random":
        blockers.append(_block("successor_initialization_not_fresh", "Fresh random initialization is required."))
    if initialization.get("resume_allowed") is not False or initialization.get("historical_state_use_allowed") is not False:
        blockers.append(_block("successor_state_reuse_open", "Resume and historical-state use must remain false."))
    for field in CANDIDATE_FORBIDDEN_STATE_INPUT_FIELDS:
        if initialization.get(field) is not None:
            blockers.append(_block("successor_historical_state_input", f"{field} must be null."))
    inherited = _inherited_protocol(config)
    predecessor = dict(contract["predecessor_authority"])
    if (
        inherited.get("candidate_lineage_uid")
        != EXPECTED_PREDECESSOR_LINEAGE_UID
    ):
        blockers.append(
            _block(
                "successor_predecessor_lineage_drift",
                "Predecessor lineage UID drifted.",
            )
        )
    if inherited.get("protocol_uid") != EXPECTED_PREDECESSOR_PROTOCOL_UID:
        blockers.append(
            _block(
                "successor_predecessor_protocol_drift",
                "Predecessor protocol UID drifted.",
            )
        )
    if (
        predecessor.get("predecessor_candidate_lineage_uid")
        != EXPECTED_PREDECESSOR_LINEAGE_UID
        or predecessor.get("predecessor_protocol_uid")
        != EXPECTED_PREDECESSOR_PROTOCOL_UID
        or predecessor.get("predecessor_preflight_uid")
        != EXPECTED_PREDECESSOR_PREFLIGHT_UID
    ):
        blockers.append(
            _block(
                "successor_predecessor_authority_drift",
                "Declared predecessor lineage, protocol, or preflight identity drifted.",
            )
        )
    if (
        canonical_hash(inherited["scientific_contract"])
        != EXPECTED_SCIENTIFIC_CONTRACT_HASH
    ):
        blockers.append(
            _block(
                "successor_scientific_contract_drift",
                "Scientific contract hash drifted.",
            )
        )
    invariant = dict(contract["invariant_contract"])
    methods = [str(row["method_id"]) for row in inherited["scientific_contract"]["methods"]]
    seeds = [int(value) for value in inherited["scientific_contract"]["seeds"]]
    if methods != REQUIRED_METHOD_IDS or methods != list(invariant["required_method_ids"]):
        blockers.append(_block("successor_method_identity_drift", "Exact five-method order is required."))
    if seeds != EXPECTED_SEEDS or seeds != list(invariant["required_seeds"]):
        blockers.append(_block("successor_seed_identity_drift", "Exact three-seed order is required."))
    if invariant.get("scientific_contract_hash") != canonical_hash(
        inherited["scientific_contract"]
    ):
        blockers.append(
            _block(
                "successor_scientific_contract_hash_mismatch",
                "Declared scientific hash does not recompute.",
            )
        )
    if invariant.get("execution_contract_hash") != canonical_hash(
        inherited["execution_contract"]
    ):
        blockers.append(
            _block(
                "successor_execution_contract_hash_mismatch",
                "Execution contract hash drifted.",
            )
        )
    if invariant.get("scheduler_evidence_contract_hash") != canonical_hash(
        inherited["scheduler_evidence_contract"]
    ):
        blockers.append(
            _block(
                "successor_scheduler_contract_hash_mismatch",
                "Scheduler contract hash drifted.",
            )
        )
    execution = dict(inherited["execution_contract"])
    attempt_slots = [dict(row) for row in execution["attempt_slots"]]
    if (
        [str(row["attempt_kind"]) for row in attempt_slots]
        != list(invariant["attempt_kinds"])
        or execution["maximum_attempts_per_assignment"]
        != invariant["maximum_attempts_per_assignment"]
        or execution["maximum_infrastructure_recovery_attempts_per_assignment"]
        != invariant["maximum_infrastructure_recovery_attempts_per_assignment"]
        or attempt_slots[1]["eligible_after_failure_class"]
        != invariant["recovery_eligible_after_failure_class"]
    ):
        blockers.append(
            _block(
                "successor_attempt_contract_drift",
                "Attempt-slot and recovery semantics must equal the repaired protocol.",
            )
        )
    resource = dict(contract["resource_contract"])
    scheduler = dict(inherited["scheduler_evidence_contract"])
    if (
        resource.get("profiles") != execution.get("profiles")
        or resource.get("wall_time_by_method")
        != execution.get("wall_time_by_method")
        or resource.get("scheduler_evidence_contract_version")
        != scheduler.get("contract_version")
        or resource.get("slurm_version") != scheduler.get("slurm_version")
        or any(
            dict(profile).get("exclusive") is not False
            for profile in dict(resource.get("profiles", {})).values()
        )
        or resource.get("fresh_sinfo_required_before_execution") is not True
        or resource.get("fresh_per_attempt_resource_receipt_required") is not True
        or resource.get("independent_execution_authority_required") is not True
        or resource.get("live_resource_receipt_created") is not False
        or resource.get("live_execution_authority_created") is not False
    ):
        blockers.append(
            _block(
                "successor_resource_contract_drift",
                "Resource and scheduler semantics must equal the repaired protocol and remain non-executing.",
            )
        )
    try:
        expected_identity = compute_successor_identity_declarations(
            config, root, pre_change_manifest
        )
    except (KeyError, TypeError, ValueError) as error:
        expected_identity = {}
        blockers.append(
            _block(
                "successor_identity_derivation_failed",
                str(error),
            )
        )
    for field, expected in expected_identity.items():
        if expected is not None and identity.get(field) != expected:
            blockers.append(
                _block(
                    "successor_identity_mismatch",
                    f"{field} does not recompute; expected {expected}.",
                )
            )
    if expected_identity and (
        dict(contract["environment_contract"]).get("environment_contract_uid")
        != expected_identity["environment_contract_uid"]
    ):
        blockers.append(
            _block(
                "successor_environment_contract_uid_mismatch",
                "Environment contract UID drifted.",
            )
        )
    if expected_identity and (
        dict(contract["resource_contract"]).get("resource_contract_uid")
        != expected_identity["resource_contract_uid"]
    ):
        blockers.append(
            _block(
                "successor_resource_contract_uid_mismatch",
                "Resource contract UID drifted.",
            )
        )
    if dict(contract["review_contract"]) != dict(schema["required_review_contract"]):
        blockers.append(_block("successor_review_contract_mismatch", "Review contract drifted."))
    terminal = dict(contract["terminal_policy"])
    if (
        terminal.get("positive") != schema.get("required_positive_terminal")
        or terminal.get("invalid") != schema.get("required_invalid_terminal")
        or terminal.get("scientific_twin_result") != schema.get("required_scientific_twin_result")
        or terminal.get("execution_handoff_required") is not True
        or terminal.get("success_artifact_on_failure") is not False
    ):
        blockers.append(_block("successor_terminal_policy_drift", "Terminal policy drifted."))
    access = dict(contract.get("access_counters", {}))
    if set(access) != set(schema.get("required_access_counter_fields", [])) or set(
        access.values()
    ) != {0}:
        blockers.append(
            _block(
                "successor_access_boundary_open",
                "All ten access counters must be exactly zero.",
            )
        )
    closed = dict(contract.get("closed_flags", {}))
    if set(closed) != set(schema.get("required_closed_flag_fields", [])) or any(
        value is not False for value in closed.values()
    ):
        blockers.append(
            _block(
                "successor_downstream_gate_open",
                "Every execution and downstream flag must remain false.",
            )
        )
    if list(publication_paths(config).values()) != list(
        schema.get("required_publication_paths", [])
    ):
        blockers.append(
            _block(
                "successor_publication_path_drift",
                "The exact seven publication paths changed.",
            )
        )
    source_identities = successor_source_identities(config, root)
    if require_committed_sources and any(row["git_object_id"] is None for row in source_identities):
        blockers.append(_block("successor_source_not_committed", "Every source binding must match a committed Git blob."))
    if any(
        isinstance(value, str) and value.startswith("__DERIVE_")
        for section in (
            identity,
            invariant,
            dict(contract["environment_contract"]),
            resource,
        )
        for value in section.values()
    ):
        blockers.append(
            _block(
                "successor_identity_placeholder_present",
                "All successor identities and invariant hashes must be materialized.",
            )
        )
    for commit in (
        predecessor["planner_commit"],
        predecessor["baseline_commit"],
        dict(predecessor["round_48"])["evidence_commit"],
        dict(predecessor["round_48"])["source_commit"],
        dict(predecessor["round_48"])["squash_merge_commit"],
        dict(predecessor["round_48"])["controller_completion_commit"],
        dict(predecessor["round_47"])["merged_commit"],
        dict(predecessor["round_47"])["evidence_commit"],
        dict(predecessor["round_47"])["source_commit"],
        dict(predecessor["round_46"])["merged_commit"],
        dict(predecessor["round_46"])["execution_source_commit"],
    ):
        try:
            _git_commit(root, str(commit))
        except OperationalTwinProtocolError as error:
            blockers.append(_block("successor_authority_commit_missing", str(error)))
    if not _is_ancestor(root, str(predecessor["baseline_commit"]), "HEAD"):
        blockers.append(_block("successor_baseline_not_ancestor", "Current source does not descend from the approved controller completion."))
    round_48 = dict(predecessor["round_48"])
    try:
        audit = _git_json(root, "HEAD", str(dict(round_48["decision_audit"])["path"]))
        if (
            audit.get("terminal_decision") != EXPECTED_ROUND_48_TERMINAL
            or audit.get("decision_uid") != EXPECTED_ROUND_48_DECISION_UID
            or audit.get("authorized_row_count") != 0
            or audit.get("successor_required_row_count") != 15
            or audit.get("candidate_dispatch_authorized") is not False
            or audit.get("scientific_twin_result") != "NOT_RUN"
            or set(dict(audit.get("access_counters", {})).values()) != {0}
        ):
            blockers.append(_block("successor_round_48_semantic_drift", "Round 48 decision semantics drifted."))
    except OperationalTwinProtocolError as error:
        blockers.append(_block("successor_round_48_authority_missing", str(error)))
    round_34 = dict(predecessor["round_34"])
    try:
        open_audit = _git_json(root, "HEAD", str(dict(round_34["audit"])["path"]))
        if (
            open_audit.get("terminal_decision") != round_34["terminal_decision"]
            or open_audit.get("training_authorized") is not True
        ):
            blockers.append(_block("successor_round_34_authority_drift", "Round 34 open-data authority drifted."))
        for role, binding_value in dict(round_34["manifests"]).items():
            binding = dict(binding_value)
            content = read_repo_bytes(root, str(binding["path"]))
            payload = json.loads(content.decode("utf-8"))
            if (
                hashlib.sha256(content).hexdigest() != binding["sha256"]
                or payload.get("content_identity") != binding["content_identity"]
            ):
                blockers.append(_block("successor_round_34_manifest_drift", role))
    except (OperationalTwinProtocolError, json.JSONDecodeError) as error:
        blockers.append(_block("successor_round_34_authority_missing", str(error)))
    if pre_change_manifest is not None:
        try:
            blockers.extend(
                validate_pre_change_identity_manifest(
                    config, root, pre_change_manifest
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            blockers.append(
                _block(
                    "successor_pre_change_identity_manifest_invalid",
                    str(error),
                )
            )
    return blockers


def _validate_future_absence(
    config: Mapping[str, object],
    root: Path,
    protocol: Mapping[str, object],
    assignments: Sequence[Mapping[str, object]],
) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    predecessor = dict(_contract(config)["predecessor_authority"])
    old_lineage = str(predecessor["predecessor_candidate_lineage_uid"])
    new_lineage = str(protocol["candidate_lineage_uid"])
    for row in _future_paths(config, protocol, assignments):
        allowed_root = (
            "Output/audits/operational_twin"
            if row["path"].startswith("Output/audits/operational_twin/")
            else "Output/results/operational_twin/candidate_lineages"
        )
        blockers.extend(
            _validate_path(
                root,
                row["path"],
                allowed_root=allowed_root,
                require_absent=True,
                require_git_absent=True,
            )
        )
        if old_lineage in row["path"] or new_lineage not in row["path"]:
            blockers.append(_block("successor_future_path_lineage_mismatch", row["path"]))
    retry_paths = dict(config["operational_twin_candidate_retry_authority"])["output_paths"]
    for key in (
        "candidate_lineage_manifest",
        "assignment_manifest",
        "assignment_commands",
        "preflight_authority",
        "preflight_report",
    ):
        value = str(dict(retry_paths)[key])
        if os.path.lexists(root / value) or _git_object(root, "HEAD", value) is not None:
            blockers.append(_block("successor_reserved_retry_path_materialized", value))
    return blockers


def _validate_commands(
    config: Mapping[str, object], built: Mapping[str, object]
) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    commands = dict(built["assignment_commands"])
    candidate = dict(built["candidate_lineage_manifest"])
    assignment_manifest = dict(built["assignment_manifest"])
    mappings = dict(built["mappings"])
    rows = commands.get("commands")
    if commands.get("schema_version") != COMMAND_SCHEMA_VERSION:
        blockers.append(
            _block("successor_command_schema_mismatch", "Command schema drifted.")
        )
    if set(commands) != SUCCESSOR_COMMAND_CATALOG_KEYS:
        blockers.append(
            _block(
                "successor_command_catalog_keyset_mismatch",
                "The successor command catalog keyset drifted.",
            )
        )
    if commands.get("command_catalog_hash") != command_catalog_hash(commands):
        blockers.append(
            _block(
                "successor_command_hash_mismatch",
                "Command catalog hash drifted.",
            )
        )
    for field in (
        "candidate_lineage_uid",
        "protocol_uid",
        "preflight_uid",
        "candidate_manifest_hash",
    ):
        if commands.get(field) != candidate.get(field):
            blockers.append(
                _block(
                    "successor_command_manifest_binding_mismatch",
                    f"Command catalog {field} drifted.",
                )
            )
    if commands.get("assignment_manifest_hash") != assignment_manifest.get(
        "assignment_manifest_hash"
    ):
        blockers.append(
            _block(
                "successor_command_assignment_binding_mismatch",
                "Command catalog assignment-manifest hash drifted.",
            )
        )
    if (
        commands.get("assignment_bijection_hash")
        != mappings.get("assignment_bijection_hash")
        or commands.get("attempt_slot_bijection_hash")
        != mappings.get("attempt_slot_bijection_hash")
    ):
        blockers.append(
            _block(
                "successor_command_mapping_binding_mismatch",
                "Command catalog predecessor mappings drifted.",
            )
        )
    if (
        commands.get("assignment_count") != 15
        or commands.get("command_count") != 30
        or commands.get("initial_command_count") != 15
        or commands.get("conditional_recovery_command_count") != 15
        or commands.get("execution_handoff_required") is not True
        or commands.get("candidate_dispatch_authorized") is not False
        or commands.get("training_executed") is not False
        or commands.get("calibration_fitting_executed") is not False
        or commands.get("checkpoint_created") is not False
    ):
        blockers.append(
            _block(
                "successor_command_closed_contract_drift",
                "Command counts and non-executing flags must remain exact.",
            )
        )
    if not isinstance(rows, list) or len(rows) != 30:
        return blockers + [
            _block(
                "successor_command_count_mismatch",
                "Exactly 30 commands are required.",
            )
        ]
    schema = _schema(config)
    if [row.get("command_index") for row in rows] != list(schema["required_command_indices"]):
        blockers.append(
            _block(
                "successor_command_index_mismatch",
                "Command indices must be 0 through 29.",
            )
        )
    candidate_uid = str(dict(built["identities"])["candidate_lineage_uid"])
    old_uid = str(
        dict(_contract(config)["predecessor_authority"])[
            "predecessor_candidate_lineage_uid"
        ]
    )
    assignments = {
        str(assignment["assignment_uid"]): dict(assignment)
        for assignment in assignment_manifest["assignments"]
    }
    runtime_config = deepcopy(dict(config))
    runtime_config["operational_twin_candidate_lineage_protocol"] = dict(
        built["protocol"]
    )
    output_roots: list[str] = []
    authorities: list[str] = []
    receipts: list[str] = []
    for index, value in enumerate(rows):
        if not isinstance(value, Mapping) or set(value) != CANDIDATE_COMMAND_ROW_KEYS:
            blockers.append(
                _block("successor_command_row_keyset_mismatch", str(index))
            )
            continue
        row = dict(value)
        argv = row.get("argv")
        assignment = assignments.get(str(row.get("assignment_uid", "")))
        attempt = None
        if assignment is not None:
            attempt = next(
                (
                    dict(item)
                    for item in assignment["attempts"]
                    if item["attempt_uid"] == row.get("attempt_uid")
                ),
                None,
            )
        if (
            row.get("candidate_lineage_uid") != candidate_uid
            or old_uid in json.dumps(row, sort_keys=True)
            or not isinstance(argv, list)
            or assignment is None
            or attempt is None
            or any(
                flag in argv
                for flag in (
                    "--resume",
                    "--resume-input",
                    "--history-input",
                    "--checkpoint-input",
                    "--optimizer-state-input",
                    "--model-state-input",
                    "--calibration-state-input",
                )
            )
        ):
            blockers.append(
                _block("successor_command_identity_or_state_drift", str(index))
            )
            continue
        expected_argv = expected_candidate_command_argv(
            runtime_config,
            candidate,
            assignment_manifest,
            assignment,
            attempt,
            command_catalog_hash_value=commands["command_catalog_hash"],
        )
        expected_fields = {
            "command_index": index,
            "candidate_lineage_uid": assignment["candidate_lineage_uid"],
            "protocol_uid": assignment["protocol_uid"],
            "assignment_uid": assignment["assignment_uid"],
            "attempt_uid": attempt["attempt_uid"],
            "attempt_kind": attempt["attempt_kind"],
            "conditional": attempt["conditional"],
            "eligible_after_failure_class": attempt[
                "eligible_after_failure_class"
            ],
            "predecessor_attempt_uid": attempt["predecessor_attempt_uid"],
            "method_id": assignment["method_id"],
            "seed": assignment["seed"],
            "resource_profile": assignment["resource_profile"],
            "resource_requirements": assignment["resource_requirements"],
            "wall_time": assignment["wall_time"],
            "output_root": attempt["output_root"],
            "execution_authority": attempt["execution_authority"],
            "resource_environment_receipt": attempt[
                "resource_environment_receipt"
            ],
            "argv": expected_argv,
        }
        if row != expected_fields:
            blockers.append(
                _block(
                    "successor_command_row_drift",
                    f"Command row {index} differs from the canonical attempt.",
                )
            )
        output_roots.append(str(row["output_root"]))
        authorities.append(str(row["execution_authority"]))
        receipts.append(str(row["resource_environment_receipt"]))
    for label, values in (
        ("output_root", output_roots),
        ("execution_authority", authorities),
        ("resource_environment_receipt", receipts),
    ):
        if len(values) != len(set(values)):
            blockers.append(
                _block(
                    "successor_command_path_collision",
                    f"Successor command {label} values are not one-to-one.",
                )
            )
    return blockers


def build_successor_preflight(
    config: Mapping[str, object],
    root: Path,
    built: Mapping[str, object],
    pre_change_manifest: Mapping[str, object],
    pre_change_identity: Mapping[str, object],
    *,
    source_commit: str,
) -> dict[str, object]:
    contract = _contract(config)
    identities = dict(built["identities"])
    candidate = dict(built["candidate_lineage_manifest"])
    assignments = dict(built["assignment_manifest"])
    commands = dict(built["assignment_commands"])
    mappings = dict(built["mappings"])
    source_identities = successor_source_identities(config, root)
    return {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "PASS",
        "terminal_decision": POSITIVE_TERMINAL,
        "authoritative_date": contract["authoritative_date"],
        "source_commit": source_commit,
        **identities,
        "candidate_manifest_hash": candidate["candidate_manifest_hash"],
        "assignment_manifest_hash": assignments["assignment_manifest_hash"],
        "command_catalog_hash": commands["command_catalog_hash"],
        "assignment_count": assignments["assignment_count"],
        "attempt_count": assignments["attempt_slot_count"],
        "command_count": commands["command_count"],
        "assignment_bijection_hash": mappings["assignment_bijection_hash"],
        "attempt_slot_bijection_hash": mappings["attempt_slot_bijection_hash"],
        "config_identity": {
            "uid": identities["config_contract_uid"],
            "basis_hash": canonical_hash(successor_config_identity_basis(config)),
            "path": "Libs/configs/exp/operational_twin_successor_lineage_protocol.yaml",
        },
        "schema_identity": {
            "uid": identities["schema_contract_uid"],
            **_file_identity(root, "Libs/configs/schema/operational_twin_successor_lineage_protocol.schema.yaml"),
        },
        "source_identities": source_identities,
        "pre_change_identity_manifest": {
            **dict(pre_change_identity),
            "manifest_hash": pre_change_manifest["manifest_hash"],
        },
        "predecessor_identity_manifest_hash": canonical_hash(
            {
                "historical_protocol_artifacts": pre_change_manifest[
                    "historical_protocol_artifacts"
                ],
                "authority_artifacts": pre_change_manifest["authority_artifacts"],
                "historical_trees": pre_change_manifest["historical_trees"],
            }
        ),
        "environment_contract": contract["environment_contract"],
        "resource_contract": contract["resource_contract"],
        "review_contract": contract["review_contract"],
        "publication_paths": publication_paths(config),
        "future_path_count": len(_future_paths(config, built["protocol"], assignments["assignments"])),
        "future_execution_paths_absent": True,
        "candidate_dispatch_authorized": False,
        "execution_handoff_required": True,
        "training_executed": False,
        "calibration_fitting_executed": False,
        "checkpoint_created": False,
        "sufficient_training": False,
        "scientific_twin_result": "NOT_RUN",
        "access_counters": contract["access_counters"],
        "closed_flags": contract["closed_flags"],
    }


def render_successor_preflight(preflight: Mapping[str, object]) -> str:
    return "\n".join(
        [
            "# Operational Twin Successor-Lineage Guarded Preflight",
            "",
            f"- Status: `{preflight['status']}`",
            f"- Terminal: `{preflight['terminal_decision']}`",
            f"- Authoritative date: `{preflight['authoritative_date']}`",
            f"- Candidate lineage UID: `{preflight['candidate_lineage_uid']}`",
            f"- Protocol UID: `{preflight['protocol_uid']}`",
            f"- Preflight UID: `{preflight['preflight_uid']}`",
            f"- Assignments: `{preflight['assignment_count'] if 'assignment_count' in preflight else 15}`",
            "- Attempt slots: `30`",
            "- Candidate dispatch authorized: `false`",
            "- Execution handoff required: `true`",
            "- Scientific Twin result: `NOT_RUN`",
            "",
            "All future attempt, authority, receipt, scheduler, log, aggregation, and catalog paths remain absent. All ten access counters remain zero.",
            "",
        ]
    )


def _artifact_content_identity(path: str, content: bytes) -> dict[str, object]:
    return {
        "path": path,
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }


def build_successor_audit(
    config: Mapping[str, object],
    built: Mapping[str, object],
    preflight: Mapping[str, object],
    pre_change_manifest: Mapping[str, object],
    validation_evidence: Mapping[str, object],
    validation_identity: Mapping[str, object],
    preliminary_contents: Mapping[str, tuple[str, bytes]],
) -> dict[str, object]:
    contract = _contract(config)
    mappings = dict(built["mappings"])
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "PASS",
        "terminal_decision": POSITIVE_TERMINAL,
        "authoritative_date": contract["authoritative_date"],
        "source_commit": preflight["source_commit"],
        **dict(built["identities"]),
        "candidate_manifest_hash": preflight["candidate_manifest_hash"],
        "assignment_manifest_hash": preflight["assignment_manifest_hash"],
        "command_catalog_hash": preflight["command_catalog_hash"],
        "assignment_bijection_hash": mappings["assignment_bijection_hash"],
        "attempt_slot_bijection_hash": mappings["attempt_slot_bijection_hash"],
        "assignment_count": 15,
        "attempt_count": 30,
        "command_count": 30,
        "predecessor_protocol_artifact_count": len(
            pre_change_manifest["historical_protocol_artifacts"]
        ),
        "predecessor_authority_artifact_count": len(
            pre_change_manifest["authority_artifacts"]
        ),
        "predecessor_tree_count": len(pre_change_manifest["historical_trees"]),
        "predecessor_pre_post_match": True,
        "assignment_bijection": mappings["assignment_bijection"],
        "attempt_slot_bijection": mappings["attempt_slot_bijection"],
        "config_identity": preflight["config_identity"],
        "schema_identity": preflight["schema_identity"],
        "source_identities": preflight["source_identities"],
        "pre_change_identity_manifest": preflight[
            "pre_change_identity_manifest"
        ],
        "predecessor_identity_manifest_hash": preflight[
            "predecessor_identity_manifest_hash"
        ],
        "environment_contract": preflight["environment_contract"],
        "resource_contract": preflight["resource_contract"],
        "review_contract": preflight["review_contract"],
        "publication_paths": preflight["publication_paths"],
        "future_path_count": preflight["future_path_count"],
        "validation_evidence": {
            **_identity_basis(validation_identity),
            "source_commit": validation_evidence["source_commit"],
            "slurm_job_id": validation_evidence.get("slurm_job_id"),
            "completed_checks": validation_evidence["completed_checks"],
        },
        "successor_artifacts": {
            key: _artifact_content_identity(path, content)
            for key, (path, content) in preliminary_contents.items()
        },
        "publication_file_count": 7,
        "future_execution_paths_absent": True,
        "candidate_dispatch_authorized": False,
        "execution_handoff_required": True,
        "training_executed": False,
        "calibration_fitting_executed": False,
        "checkpoint_created": False,
        "sufficient_training": False,
        "presealed_freeze_created": False,
        "scientific_twin_result": "NOT_RUN",
        "access_counters": contract["access_counters"],
        "closed_flags": contract["closed_flags"],
        "blockers": [],
    }


def render_successor_report(audit: Mapping[str, object]) -> str:
    return "\n".join(
        [
            "# Operational Twin Successor-Lineage Protocol",
            "",
            f"- Status: `{audit['status']}`",
            f"- Terminal: `{audit['terminal_decision']}`",
            f"- Source commit: `{audit['source_commit']}`",
            f"- Candidate lineage UID: `{audit['candidate_lineage_uid']}`",
            f"- Protocol UID: `{audit['protocol_uid']}`",
            f"- Preflight UID: `{audit['preflight_uid']}`",
            f"- Assignments / attempts / commands: `{audit['assignment_count']} / {audit['attempt_count']} / {audit['command_count']}`",
            f"- Predecessor protocol artifacts preserved: `{audit['predecessor_protocol_artifact_count']}`",
            f"- Predecessor authority artifacts preserved: `{audit['predecessor_authority_artifact_count']}`",
            "- Candidate dispatch authorized: `false`",
            "- Execution handoff required: `true`",
            "- Training, calibration fitting, checkpoint creation: `false`",
            "- Scientific Twin result: `NOT_RUN`",
            "",
            "The successor is a frozen, non-executing protocol only. Every Round 46 artifact remains historical evidence and is neither an initialization input nor successor evidence. All ten access counters remain zero.",
            "",
        ]
    )


def validate_source_gate_evidence(
    config: Mapping[str, object],
    root: Path,
    evidence: Mapping[str, object],
    *,
    current_source_commit: str,
) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    identity = dict(_contract(config)["identity"])
    expected = {
        "schema_version": SOURCE_GATE_SCHEMA_VERSION,
        "status": "PASS",
        "terminal_decision": POSITIVE_TERMINAL,
        "authoritative_date": EXPECTED_AUTHORITATIVE_DATE,
        "source_commit": current_source_commit,
        "clean_tracked_source": True,
        "completed_checks": EXPECTED_SOURCE_GATE_CHECKS,
        "candidate_lineage_uid": identity["candidate_lineage_uid"],
        "protocol_uid": identity["protocol_uid"],
        "preflight_uid": identity["preflight_uid"],
        "candidate_dispatch_authorized": False,
        "scientific_twin_result": "NOT_RUN",
        "access_counters": _contract(config)["access_counters"],
    }
    for field, value in expected.items():
        if evidence.get(field) != value:
            blockers.append(_block("successor_source_gate_invalid", f"{field} drifted."))
    if not isinstance(evidence.get("slurm_job_id"), str) or not str(
        evidence.get("slurm_job_id")
    ).isdigit():
        blockers.append(
            _block(
                "successor_source_gate_job_id_invalid",
                "A Slurm job ID is required.",
            )
        )
    try:
        _git_commit(root, current_source_commit)
        if not _is_ancestor(root, current_source_commit, "HEAD"):
            blockers.append(
                _block(
                    "successor_source_gate_commit_not_ancestor",
                    "The validated source commit is not an ancestor of current HEAD.",
                )
            )
        for identity in successor_source_identities(config, root):
            path = str(identity["path"])
            if _git_bytes(root, current_source_commit, path) != read_repo_bytes(
                root, path
            ):
                blockers.append(
                    _block(
                        "successor_source_gate_binding_drift",
                        f"{path} differs from the validated source commit.",
                    )
                )
    except OperationalTwinProtocolError as error:
        blockers.append(_block("successor_source_gate_commit_invalid", str(error)))
    return blockers


def build_source_gate_evidence(
    config: Mapping[str, object],
    root: Path,
    *,
    completed_checks: Sequence[str],
    source_commit: str,
    slurm_job_id: str,
) -> dict[str, object]:
    identity = dict(_contract(config)["identity"])
    return {
        "schema_version": SOURCE_GATE_SCHEMA_VERSION,
        "status": "PASS",
        "terminal_decision": POSITIVE_TERMINAL,
        "authoritative_date": EXPECTED_AUTHORITATIVE_DATE,
        "source_commit": source_commit,
        "slurm_job_id": slurm_job_id,
        "clean_tracked_source": True,
        "completed_checks": sorted(completed_checks),
        "candidate_lineage_uid": identity["candidate_lineage_uid"],
        "protocol_uid": identity["protocol_uid"],
        "preflight_uid": identity["preflight_uid"],
        "candidate_dispatch_authorized": False,
        "scientific_twin_result": "NOT_RUN",
        "access_counters": _contract(config)["access_counters"],
    }


def build_successor_package(
    config: Mapping[str, object],
    root: Path,
    *,
    pre_change_manifest_path: Path | None = None,
    validation_evidence_path: Path | None = None,
    source_commit: str | None = None,
    require_publication_targets_absent: bool,
    require_committed_sources: bool,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, str]]]:
    blockers: list[dict[str, str]] = []
    pre_change_manifest, pre_change_identity = _read_pre_change_manifest(
        config, root, pre_change_manifest_path
    )
    if require_committed_sources and pre_change_identity.get("git_object_id") is None:
        blockers.append(
            _block(
                "successor_pre_change_identity_manifest_not_committed",
                "The pre-change identity manifest must match a committed Git blob.",
            )
        )
    blockers.extend(
        validate_successor_config(
            config,
            root,
            pre_change_manifest=pre_change_manifest,
            require_committed_sources=require_committed_sources,
        )
    )
    built = build_successor_artifacts(config, root, pre_change_manifest)
    blockers.extend(
        _validate_bijections(
            built,
            str(dict(_contract(config)["predecessor_authority"])["predecessor_candidate_lineage_uid"]),
        )
    )
    blockers.extend(_validate_commands(config, built))
    protocol = dict(built["protocol"])
    assignments = list(dict(built["assignment_manifest"])["assignments"])
    blockers.extend(_validate_future_absence(config, root, protocol, assignments))
    if require_publication_targets_absent:
        protocol_root = str(
            dict(_contract(config)["output_namespace"])["protocol_root"]
        )
        blockers.extend(
            _validate_path(
                root,
                protocol_root,
                allowed_root="Output/results/operational_twin",
                require_absent=True,
                require_git_absent=True,
            )
        )
        for role, value in publication_paths(config).items():
            allowed_root = (
                "Output/results/operational_twin"
                if value.startswith("Output/results/")
                else "Output/audits"
                if value.startswith("Output/audits/")
                else "Output/reports"
            )
            blockers.extend(
                _validate_path(
                    root,
                    value,
                    allowed_root=allowed_root,
                    require_absent=True,
                    require_git_absent=True,
                )
            )
    effective_source_commit = source_commit or _current_commit(root)
    preflight = build_successor_preflight(
        config,
        root,
        built,
        pre_change_manifest,
        pre_change_identity,
        source_commit=effective_source_commit,
    )
    preliminary = {
        "candidate_lineage_manifest": (
            publication_paths(config)["candidate_lineage_manifest"],
            _json_bytes(built["candidate_lineage_manifest"]),
        ),
        "assignment_manifest": (
            publication_paths(config)["assignment_manifest"],
            _json_bytes(built["assignment_manifest"]),
        ),
        "assignment_commands": (
            publication_paths(config)["assignment_commands"],
            _json_bytes(built["assignment_commands"]),
        ),
        "preflight_authority": (
            publication_paths(config)["preflight_authority"],
            _json_bytes(preflight),
        ),
        "preflight_report": (
            publication_paths(config)["preflight_report"],
            render_successor_preflight(preflight).encode("utf-8"),
        ),
    }
    validation_evidence: dict[str, object]
    validation_identity: dict[str, object]
    if validation_evidence_path is None:
        validation_evidence = {
            "source_commit": effective_source_commit,
            "slurm_job_id": None,
            "completed_checks": EXPECTED_SOURCE_GATE_CHECKS,
        }
        validation_identity = {
            "path": str(dict(_contract(config)["evidence_inputs"])["validation_evidence"]),
            "sha256": None,
            "size_bytes": 0,
            "git_object_id": None,
        }
    else:
        validation_evidence, validation_identity = _read_validation_evidence(
            config, root, validation_evidence_path
        )
        blockers.extend(
            validate_source_gate_evidence(
                config,
                root,
                validation_evidence,
                current_source_commit=effective_source_commit,
            )
        )
        configured_pre_change = str(
            dict(_contract(config)["evidence_inputs"])[
                "pre_change_identity_manifest"
            ]
        )
        source_manifest_object = _git_object(
            root, effective_source_commit, configured_pre_change
        )
        if (
            source_manifest_object is None
            or source_manifest_object != pre_change_identity.get("git_object_id")
        ):
            blockers.append(
                _block(
                    "successor_pre_change_source_binding_mismatch",
                    "The pre-change manifest is not the exact blob validated at the source commit.",
                )
            )
    audit = build_successor_audit(
        config,
        built,
        preflight,
        pre_change_manifest,
        validation_evidence,
        validation_identity,
        preliminary,
    )
    package = {
        **preliminary,
        "protocol_audit": (
            publication_paths(config)["protocol_audit"], _json_bytes(audit)
        ),
        "protocol_report": (
            publication_paths(config)["protocol_report"],
            render_successor_report(audit).encode("utf-8"),
        ),
    }
    predecessor_identities = [
        *pre_change_manifest["historical_protocol_artifacts"],
        *pre_change_manifest["authority_artifacts"],
    ]
    return package, predecessor_identities, blockers


def validate_materialized_successor_package(
    config: Mapping[str, object], root: Path
) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    paths = publication_paths(config)
    loaded: dict[str, tuple[str, bytes]] = {}
    for key, path in paths.items():
        allowed_root = (
            "Output/results/operational_twin"
            if path.startswith("Output/results/")
            else "Output/audits"
            if path.startswith("Output/audits/")
            else "Output/reports"
        )
        blockers.extend(
            _validate_path(
                root,
                path,
                allowed_root=allowed_root,
                require_absent=False,
                require_git_absent=False,
            )
        )
        target = root / path
        if not target.is_file() or target.is_symlink():
            blockers.append(_block("successor_publication_file_missing", path))
            continue
        content = target.read_bytes()
        if key in JSON_PUBLICATION_KEYS:
            try:
                payload = json.loads(content.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                blockers.append(_block("successor_publication_json_invalid", path))
                continue
            if not isinstance(payload, dict) or content != _json_bytes(payload):
                blockers.append(_block("successor_publication_json_noncanonical", path))
                continue
        loaded[key] = (path, content)
    if blockers:
        return blockers
    audit = json.loads(loaded["protocol_audit"][1].decode("utf-8"))
    source_commit = str(audit.get("source_commit", ""))
    try:
        expected, predecessor_identities, recompute_blockers = build_successor_package(
            config,
            root,
            validation_evidence_path=root
            / str(dict(_contract(config)["evidence_inputs"])["validation_evidence"]),
            source_commit=source_commit,
            require_publication_targets_absent=False,
            require_committed_sources=True,
        )
        blockers.extend(recompute_blockers)
        for key in PUBLICATION_KEYS:
            if loaded[key][1] != expected[key][1]:
                blockers.append(_block("successor_materialized_artifact_drift", key))
        blockers.extend(validate_preserved_identities(root, predecessor_identities))
        protocol = derive_successor_protocol(
            config,
            root,
            identities={key: str(value) for key, value in dict(_contract(config)["identity"]).items()},
        )
        assignments = build_candidate_assignments(protocol)
        blockers.extend(_validate_future_absence(config, root, protocol, assignments))
    except (OperationalTwinProtocolError, KeyError, TypeError, ValueError) as error:
        blockers.append(_block("successor_materialized_recomputation_failed", str(error)))
    protocol_root = root / str(dict(_contract(config)["output_namespace"])["protocol_root"])
    blockers.extend(
        _validate_path(
            root,
            str(dict(_contract(config)["output_namespace"])["protocol_root"]),
            allowed_root="Output/results/operational_twin",
            require_absent=False,
            require_git_absent=False,
        )
    )
    if protocol_root.is_symlink():
        blockers.append(
            _block(
                "successor_protocol_root_symlink",
                str(protocol_root),
            )
        )
    elif protocol_root.is_dir():
        names = sorted(path.name for path in protocol_root.iterdir())
        if names != [
            "assignment_commands.json",
            "assignment_manifest.json",
            "candidate_lineage_manifest.json",
        ]:
            blockers.append(_block("successor_protocol_root_contents_invalid", str(names)))
    return blockers


def validate_preserved_identities(
    root: Path, identities: Sequence[Mapping[str, object]]
) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    for identity in identities:
        path = str(identity["path"])
        content = read_repo_bytes(root, path)
        if (
            len(content) != int(identity["size_bytes"])
            or hashlib.sha256(content).hexdigest() != identity["sha256"]
            or _git_object(root, "HEAD", path) != identity["git_object_id"]
        ):
            blockers.append(_block("successor_predecessor_identity_drift", path))
    return blockers


def package_binding_uid(artifact_identities: Mapping[str, object]) -> str:
    return _uid("operational_twin_successor_review_binding", artifact_identities)


def require_approved_successor_lineage_protocol(
    config: Mapping[str, object], root: Path
) -> dict[str, object]:
    blockers = validate_materialized_successor_package(config, root)
    if blockers:
        raise OperationalTwinProtocolError(
            "successor_reviewed_package_invalid: " + json.dumps(blockers, sort_keys=True)
        )
    artifacts: dict[str, object] = {}
    package: dict[str, object] = {}
    for key, path in publication_paths(config).items():
        content = _git_bytes(root, "HEAD", path)
        actual = (root / path).read_bytes()
        if actual != content:
            raise OperationalTwinProtocolError(
                f"successor_committed_artifact_mismatch: {path}"
            )
        artifacts[key] = _git_identity(root, "HEAD", path)
        package[key] = (
            json.loads(content.decode("utf-8"))
            if key in JSON_PUBLICATION_KEYS
            else content.decode("utf-8")
        )
    review = dict(_contract(config)["review_contract"])
    review_path = str(review["path"])
    review_content = _git_bytes(root, "HEAD", review_path)
    review_target = root / review_path
    if (
        not review_target.is_file()
        or review_target.is_symlink()
        or review_target.read_bytes() != review_content
    ):
        raise OperationalTwinProtocolError(
            "successor_committed_review_mismatch"
        )
    try:
        review_payload = json.loads(review_content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OperationalTwinProtocolError(
            "successor_review_json_invalid"
        ) from error
    if not isinstance(review_payload, dict):
        raise OperationalTwinProtocolError("successor_review_json_invalid")
    if set(review_payload) != REVIEW_RECORD_KEYS:
        raise OperationalTwinProtocolError("successor_review_keyset_mismatch")
    for field, expected in review.items():
        if field == "path":
            continue
        if review_payload.get(field) != expected:
            raise OperationalTwinProtocolError(
                f"successor_review_{field}_mismatch"
            )
    binding_uid = package_binding_uid(artifacts)
    summary = review_payload.get("evidence_summary")
    if (
        not isinstance(summary, str)
        or POSITIVE_TERMINAL not in summary
        or str(dict(_contract(config)["identity"])["candidate_lineage_uid"])
        not in summary
        or binding_uid not in summary
    ):
        raise OperationalTwinProtocolError("successor_review_binding_mismatch")
    return {
        "schema_version": PACKAGE_GATE_SCHEMA_VERSION,
        "status": "PASS",
        "terminal_decision": POSITIVE_TERMINAL,
        "candidate_lineage_uid": dict(_contract(config)["identity"])[
            "candidate_lineage_uid"
        ],
        "protocol_uid": dict(_contract(config)["identity"])["protocol_uid"],
        "preflight_uid": dict(_contract(config)["identity"])["preflight_uid"],
        "artifact_identities": artifacts,
        "review_identity": _git_identity(root, "HEAD", review_path),
        "review_binding_uid": binding_uid,
        "candidate_dispatch_authorized": False,
        "execution_handoff_required": True,
        "execution_authority": None,
        "resource_environment_receipt": None,
        "scientific_twin_result": "NOT_RUN",
        "access_counters": _contract(config)["access_counters"],
        "package": package,
    }
