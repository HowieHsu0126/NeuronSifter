"""Build structural no-task-fit route arbitration artifacts.

The builder reads only the frozen parent authority and the explicit asset
allowlist. It does not execute simulators, load checkpoints, run inference,
fit parameters, inspect evaluation outcomes, or access protected data.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = "Libs/configs/validation/no_task_fit_route_arbitration.yaml"
EXPECTED_CONTRACT_FILE_SHA256 = (
    "cc28597bc86eb266483cabda3c1dbe5b31ca8a4dc7b5b999a96e2d1fb3ede05a"
)
PARENT_PATH = "Output/audits/no_task_fit_parent_authority.json"
EXPECTED_PARENT_SHA256 = (
    "56eea3c26d292d2c2e5575cf89021bdeea4055bd6c66dfadcbcb3c1a3e86f554"
)

LINEAGE_OUTPUT = "Output/audits/no_task_fit_reusable_asset_lineage.json"
ARBITRATION_OUTPUT = "Output/audits/no_task_fit_route_arbitration.json"
CANDIDATE_OUTPUT = "Output/audits/no_task_fit_candidate_registry.json"

EXPECTED_OUTPUTS = {
    "reusable_asset_lineage": {
        "path": LINEAGE_OUTPUT,
        "schema_path": "Libs/configs/schema/no_task_fit_reusable_asset_lineage.schema.json",
        "schema_version": "neuronsifter-no-task-fit-reusable-asset-lineage-v1",
        "status": "NO_TASK_FIT_REUSABLE_ASSET_LINEAGE_BOUND",
    },
    "route_arbitration": {
        "path": ARBITRATION_OUTPUT,
        "schema_path": "Libs/configs/schema/no_task_fit_route_arbitration.schema.json",
        "schema_version": "neuronsifter-no-task-fit-route-arbitration-v1",
    },
    "candidate_registry": {
        "path": CANDIDATE_OUTPUT,
        "schema_path": "Libs/configs/schema/no_task_fit_candidate_registry.schema.json",
        "schema_version": "neuronsifter-no-task-fit-candidate-registry-v1",
    },
}

EXPECTED_ASSET_PATHS = (
    "Output/audits/no_task_fit_parent_authority.json",
    "Libs/models/modules/neuronsifter_world_action.py",
    "Libs/configs/schema/evidence_support_domain.schema.json",
    "Libs/configs/model/dynamics_solver.yaml",
    "Libs/configs/runtime/osp_pksim_environment.yaml",
    "Libs/scripts/teacher_generation/run_neuron_rxd_microenvironment_teacher.py",
    "Libs/scripts/teacher_generation/run_osp_pksim_rollout.R",
    "Libs/scripts/builders/build_full_microenvironment_teacher_registry.py",
    "Libs/scripts/validators/validate_full_microenvironment_teacher_registry.py",
    "Libs/scripts/builders/build_simulation_teacher_ready_registry.py",
    "Libs/scripts/validators/validate_simulation_teacher_ready_registry.py",
    "Libs/scripts/validators/validate_mechanistic_teacher_runtime.py",
    "Output/audits/frozen_zero_shot_method_authority.json",
    "Output/audits/frozen_zero_shot_method_manifest.json",
    "Output/audits/zero_shot_protocol_executability_decision.json",
    "Output/runs/round23/c1/large_seed3407/last.pt",
    "Output/runs/round23/c2s/large_seed3407/last.pt",
    "Libs/utils/data/structural_prior_registry.py",
    "Libs/scripts/builders/build_structural_prior_registry.py",
    "Libs/scripts/validators/validate_structural_prior_registry.py",
    "Libs/scripts/builders/build_atlas_prior_registry.py",
    "Libs/scripts/validators/validate_atlas_prior_registry.py",
    "Libs/scripts/builders/build_pkpd_ready_registry.py",
    "Libs/scripts/validators/validate_pkpd_ready_registry.py",
)
EXPECTED_ABSENT_PATHS = {
    "Output/runs/round23/c1/large_seed3407/last.pt",
    "Output/runs/round23/c2s/large_seed3407/last.pt",
}
ROUTE_IDS = ("route_a", "route_b", "route_c", "route_d")
RUBRIC_IDS = (
    "provenance_license_clarity",
    "complete_method_identity",
    "no_task_fit_feasibility",
    "interface_compatibility",
    "modality_control_coverage",
    "deterministic_or_seed_controlled_execution",
    "uncertainty_semantics",
    "execution_resource_feasibility",
    "scientific_limitations_bound",
)
ALLOWED_RESULTS = {"satisfied", "unsatisfied", "not_applicable"}
ALLOWED_DISPOSITIONS = {"admissible", "inadmissible", "not_available"}
ALLOWED_ROUTE_ROLES = {"shared", *ROUTE_IDS}
ALLOWED_ASSET_KINDS = {
    "authority_manifest",
    "historical_authority",
    "model_module",
    "schema",
    "model_configuration",
    "runtime_configuration",
    "execution_script",
    "registry_builder",
    "registry_validator",
    "checkpoint_requirement",
    "prior_module",
}
CLASSIFICATION_KEYS = {
    "executable_code",
    "configuration",
    "schema",
    "historical_authority",
    "explicitly_absent_required_identity",
}
ACCESS_BOUNDARY = {
    "six_context_outcome_access": 0,
    "protected_holdout_access": 0,
    "new_intake_performed": False,
    "task_specific_fitting_performed": False,
    "simulator_execution_performed": False,
    "candidate_inference_performed": False,
    "six_context_files_opened": False,
    "protected_files_opened": False,
}
AUTHORIZATION_CLOSURE = {
    "new_public_data_intake_authorized": False,
    "training_authorized": False,
    "gpu_authorized": False,
    "world_action_compiler_authorized": False,
    "uncertainty_implementation_authorized": False,
    "formal_evaluation_authorized": False,
    "six_context_evaluation_authorized": False,
    "tool_calling_authorized": False,
    "memory_adaptation_authorized": False,
    "agent_in_twin_authorized": False,
    "coevolution_authorized": False,
}
TERMINAL_DECISIONS = {
    "formal_zero_shot_checkpoint": "NO_FORMAL_ZERO_SHOT_CHECKPOINT",
    "prospective_training_domain": "NO_INDEPENDENT_PROSPECTIVE_TRAINING_DOMAIN",
    "public_development_data": "PUBLIC_PROSPECTIVE_DEVELOPMENT_DATA_INSUFFICIENT",
}
PARENT_AGGREGATE_ACCESS = {
    "development_outcome_access": 0,
    "evaluation_outcome_access": 0,
    "numeric_outcome_retrieval_observed": False,
    "protected_files_opened": False,
    "protected_holdout_access": 0,
    "protected_test_access": 0,
    "six_context_files_opened": False,
    "six_context_outcome_access": 0,
}
PARENT_AUTHORIZATION_CLOSURE = {
    "agent_in_twin_authorized": False,
    "automatic_second_intake_allowed": False,
    "coevolution_authorized": False,
    "formal_evaluation_authorized": False,
    "gpu_authorized": False,
    "memory_adaptation_authorized": False,
    "new_public_data_intake_authorized": False,
    "six_context_evaluation_authorized": False,
    "tool_calling_authorized": False,
    "training_authorized": False,
    "tri_authorized": False,
    "world_action_compiler_authorized": False,
}
FORBIDDEN_OUTPUT_KEYS = {
    "metric",
    "metrics",
    "score",
    "rank",
    "ranking",
    "weight",
    "weights",
    "winner",
    "preferred",
    "preferred_route",
    "default_route",
    "efficacy",
    "evaluation_result",
    "inference_result",
    "prediction",
}
CONTRACT_KEYS = {
    "schema_version",
    "contract_uid",
    "parent_authority",
    "outputs",
    "rubric",
    "disposition_rules",
    "access_boundary",
    "authorization_closure",
    "assets",
    "routes",
}


class RouteArbitrationError(ValueError):
    """Raised when route arbitration cannot be reproduced safely."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(dict(value), indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")


def canonical_value_sha256(value: object) -> str:
    return sha256_bytes(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    )


def _require_mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise RouteArbitrationError(f"{label} must be an object with string keys")
    return value


def _validate_relative_file_path(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RouteArbitrationError(
            f"{label} must be a non-empty repository-relative file path"
        )
    if any(character in value for character in "*?[]\x00"):
        raise RouteArbitrationError(
            f"{label} contains a glob or NUL character: {value!r}"
        )
    pure = PurePosixPath(value)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or "." in pure.parts
        or str(pure) != value
    ):
        raise RouteArbitrationError(
            f"{label} is not a normalized repository-relative path: {value!r}"
        )
    if not pure.suffix:
        raise RouteArbitrationError(
            f"{label} must identify a file, not a directory: {value!r}"
        )
    return value


def _resolve_path(
    root: Path, relative: str, *, expected_presence: str, label: str
) -> Path:
    normalized = _validate_relative_file_path(relative, label=label)
    resolved_root = root.resolve(strict=True)
    candidate = resolved_root.joinpath(*PurePosixPath(normalized).parts)
    current = resolved_root
    for part in PurePosixPath(normalized).parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise RouteArbitrationError(f"{label} traverses a symlink: {relative}")
    if expected_presence == "absent":
        if candidate.exists() or candidate.is_symlink():
            raise RouteArbitrationError(f"expected-absent asset is present: {relative}")
        resolved_parent = candidate.parent.resolve(strict=False)
        if not resolved_parent.is_relative_to(resolved_root):
            raise RouteArbitrationError(f"{label} escapes repository root: {relative}")
        return candidate
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise RouteArbitrationError(
            f"missing expected-present asset: {relative}"
        ) from exc
    if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
        raise RouteArbitrationError(
            f"{label} is not a regular in-repository file: {relative}"
        )
    return candidate


def _load_contract(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    observed_hash = sha256_bytes(raw)
    if observed_hash != EXPECTED_CONTRACT_FILE_SHA256:
        raise RouteArbitrationError(
            f"frozen route-arbitration contract drift: expected {EXPECTED_CONTRACT_FILE_SHA256}, observed {observed_hash}"
        )
    try:
        payload = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as exc:
        raise RouteArbitrationError(f"invalid route-arbitration YAML: {exc}") from exc
    return _require_mapping(payload, label="route-arbitration contract"), raw


def _check_exact_keys(
    value: Mapping[str, Any], expected: set[str], *, label: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise RouteArbitrationError(
            f"{label} fields differ from contract; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _validate_contract(contract: Mapping[str, Any]) -> None:
    _check_exact_keys(contract, CONTRACT_KEYS, label="contract")
    if (
        contract["schema_version"]
        != "neuronsifter-no-task-fit-route-arbitration-contract-v1"
    ):
        raise RouteArbitrationError("unexpected contract schema_version")
    if contract["contract_uid"] != "no_task_fit_route_arbitration":
        raise RouteArbitrationError("unexpected contract_uid")

    parent = _require_mapping(contract["parent_authority"], label="parent_authority")
    _check_exact_keys(
        parent,
        {
            "path",
            "sha256",
            "schema_version",
            "status",
            "artifact_count",
            "required_fact_count",
            "validator_path",
            "contract_path",
        },
        label="parent_authority",
    )
    if parent != {
        "path": PARENT_PATH,
        "sha256": EXPECTED_PARENT_SHA256,
        "schema_version": "neuronsifter-no-task-fit-parent-authority-v1",
        "status": "NO_TASK_FIT_PARENT_AUTHORITIES_BOUND",
        "artifact_count": 9,
        "required_fact_count": 101,
        "validator_path": "Libs/scripts/validators/validate_no_task_fit_parent_authority.py",
        "contract_path": "Libs/configs/validation/no_task_fit_parent_authority.yaml",
    }:
        raise RouteArbitrationError("parent authority identity drift")
    if contract["outputs"] != EXPECTED_OUTPUTS:
        raise RouteArbitrationError("output identities differ from the frozen contract")
    if contract["access_boundary"] != ACCESS_BOUNDARY:
        raise RouteArbitrationError("access boundary is not closed")
    if contract["authorization_closure"] != AUTHORIZATION_CLOSURE:
        raise RouteArbitrationError("downstream authorization closure drift")

    rubric = contract["rubric"]
    if not isinstance(rubric, list) or len(rubric) != len(RUBRIC_IDS):
        raise RouteArbitrationError("rubric must contain exactly nine criteria")
    if [row.get("criterion_id") for row in rubric if isinstance(row, dict)] != list(
        RUBRIC_IDS
    ):
        raise RouteArbitrationError("rubric criterion identity or order drift")
    for index, raw in enumerate(rubric):
        row = _require_mapping(raw, label=f"rubric[{index}]")
        _check_exact_keys(
            row, {"criterion_id", "mandatory", "description"}, label=f"rubric[{index}]"
        )
        if (
            row["mandatory"] is not True
            or not isinstance(row["description"], str)
            or not row["description"]
        ):
            raise RouteArbitrationError(
                f"invalid mandatory rubric criterion: {row['criterion_id']}"
            )

    rules = _require_mapping(contract["disposition_rules"], label="disposition_rules")
    _check_exact_keys(
        rules,
        {
            "allowed_results",
            "allowed_dispositions",
            "admissible_rule",
            "inadmissible_rule",
            "not_available_rule",
            "candidate_projection",
            "empty_registry_terminal",
            "forbidden_field_names",
        },
        label="disposition_rules",
    )
    if rules["allowed_results"] != ["satisfied", "unsatisfied", "not_applicable"]:
        raise RouteArbitrationError("criterion result vocabulary drift")
    if rules["allowed_dispositions"] != ["admissible", "inadmissible", "not_available"]:
        raise RouteArbitrationError("route disposition vocabulary drift")
    if rules["candidate_projection"] != "all_and_only_admissible_routes":
        raise RouteArbitrationError("candidate projection rule drift")
    if rules["empty_registry_terminal"] != "NO_ADMISSIBLE_NO_TASK_FIT_TWIN_ROUTE":
        raise RouteArbitrationError("empty candidate-registry terminal drift")
    if set(rules["forbidden_field_names"]) != FORBIDDEN_OUTPUT_KEYS:
        raise RouteArbitrationError("forbidden output field set drift")

    assets = contract["assets"]
    if not isinstance(assets, list) or len(assets) != len(EXPECTED_ASSET_PATHS):
        raise RouteArbitrationError("asset allowlist must contain exactly 24 entries")
    if [row.get("path") for row in assets if isinstance(row, dict)] != list(
        EXPECTED_ASSET_PATHS
    ):
        raise RouteArbitrationError("asset path identity or order drift")
    seen_uids: set[str] = set()
    seen_paths: set[str] = set()
    for index, raw in enumerate(assets):
        asset = _require_mapping(raw, label=f"assets[{index}]")
        _check_exact_keys(
            asset,
            {
                "asset_uid",
                "path",
                "route_roles",
                "asset_kind",
                "provenance_role",
                "license_access_evidence_status",
                "expected_presence",
                "sha256",
                "bytes",
                "git_blob_sha1",
                "git_tracking_state",
                "classification",
                "required_text_markers",
                "limitations",
            },
            label=f"assets[{index}]",
        )
        uid = asset["asset_uid"]
        if (
            not isinstance(uid, str)
            or re.fullmatch(r"[a-z][a-z0-9_]*", uid) is None
            or uid in seen_uids
        ):
            raise RouteArbitrationError(f"invalid or duplicate asset_uid: {uid!r}")
        seen_uids.add(uid)
        path = _validate_relative_file_path(
            asset["path"], label=f"assets[{index}].path"
        )
        if path in seen_paths:
            raise RouteArbitrationError(f"duplicate asset path: {path}")
        seen_paths.add(path)
        roles = asset["route_roles"]
        if (
            not isinstance(roles, list)
            or not roles
            or len(roles) != len(set(roles))
            or not set(roles) <= ALLOWED_ROUTE_ROLES
        ):
            raise RouteArbitrationError(f"invalid route_roles for {path}")
        if asset["asset_kind"] not in ALLOWED_ASSET_KINDS:
            raise RouteArbitrationError(f"unknown asset_kind for {path}")
        classification = _require_mapping(
            asset["classification"], label=f"{path}.classification"
        )
        _check_exact_keys(
            classification, CLASSIFICATION_KEYS, label=f"{path}.classification"
        )
        if not all(type(value) is bool for value in classification.values()):
            raise RouteArbitrationError(
                f"asset classification values must be booleans: {path}"
            )
        markers = asset["required_text_markers"]
        if (
            not isinstance(markers, list)
            or len(markers) != len(set(markers))
            or not all(isinstance(item, str) and item for item in markers)
        ):
            raise RouteArbitrationError(f"invalid required_text_markers for {path}")
        limitations = asset["limitations"]
        if (
            not isinstance(limitations, list)
            or not limitations
            or not all(isinstance(item, str) and item for item in limitations)
        ):
            raise RouteArbitrationError(f"invalid limitations for {path}")
        presence = asset["expected_presence"]
        if path in EXPECTED_ABSENT_PATHS:
            if presence != "absent" or any(
                asset[key] is not None for key in ("sha256", "bytes", "git_blob_sha1")
            ):
                raise RouteArbitrationError(
                    f"expected-absent asset identity is not null: {path}"
                )
            if asset["git_tracking_state"] != "absent" or markers:
                raise RouteArbitrationError(
                    f"expected-absent asset state is invalid: {path}"
                )
            if classification["explicitly_absent_required_identity"] is not True:
                raise RouteArbitrationError(
                    f"absent checkpoint is not classified explicitly: {path}"
                )
        else:
            if presence != "present":
                raise RouteArbitrationError(
                    f"expected-present asset is marked absent: {path}"
                )
            if (
                not isinstance(asset["sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", asset["sha256"]) is None
            ):
                raise RouteArbitrationError(f"invalid asset SHA-256: {path}")
            if type(asset["bytes"]) is not int or asset["bytes"] <= 0:
                raise RouteArbitrationError(f"invalid asset byte count: {path}")
            if (
                not isinstance(asset["git_blob_sha1"], str)
                or re.fullmatch(r"[0-9a-f]{40}", asset["git_blob_sha1"]) is None
            ):
                raise RouteArbitrationError(f"invalid asset Git blob: {path}")
            if asset["git_tracking_state"] != "tracked_clean":
                raise RouteArbitrationError(
                    f"expected-present asset must be tracked_clean: {path}"
                )

    routes = contract["routes"]
    if not isinstance(routes, list) or len(routes) != len(ROUTE_IDS):
        raise RouteArbitrationError("contract must define Routes A-D exactly once")
    if [row.get("route_id") for row in routes if isinstance(row, dict)] != list(
        ROUTE_IDS
    ):
        raise RouteArbitrationError("route identity or order drift")
    for index, raw in enumerate(routes):
        route = _require_mapping(raw, label=f"routes[{index}]")
        _check_exact_keys(
            route,
            {
                "route_id",
                "route_name",
                "scientific_label",
                "availability_state",
                "availability_reason",
                "required_asset_paths",
                "depends_on_route_ids",
                "combination_rule_status",
                "capability_support_summary",
                "assumptions",
                "criteria",
                "limitations",
                "prohibited_claims",
                "expected_disposition",
            },
            label=f"routes[{index}]",
        )
        required_paths = route["required_asset_paths"]
        evidence_paths = [
            path
            for criterion in route["criteria"]
            for path in criterion["evidence_paths"]
        ]
        for path in [*required_paths, *evidence_paths]:
            _validate_relative_file_path(
                path, label=f"{route['route_id']}.evidence_path"
            )
            if path not in seen_paths:
                raise RouteArbitrationError(
                    f"route references non-allowlisted asset: {path}"
                )
        if len(required_paths) != len(set(required_paths)):
            raise RouteArbitrationError(
                f"duplicate required asset path for {route['route_id']}"
            )
        if route["availability_state"] not in {"available", "not_available"}:
            raise RouteArbitrationError(
                f"invalid availability state for {route['route_id']}"
            )
        if (
            not isinstance(route["availability_reason"], str)
            or not route["availability_reason"]
        ):
            raise RouteArbitrationError(
                f"missing availability reason for {route['route_id']}"
            )
        capability = _require_mapping(
            route["capability_support_summary"],
            label=f"{route['route_id']}.capability_support_summary",
        )
        _check_exact_keys(
            capability,
            {
                "model_family",
                "action_modalities",
                "control_modes",
                "input_interface_status",
                "output_interface_status",
                "parameter_identity_status",
                "solver_environment_status",
                "execution_policy",
                "uncertainty_hook_status",
            },
            label=f"{route['route_id']}.capability_support_summary",
        )
        scalar_capability_fields = {
            "model_family",
            "input_interface_status",
            "output_interface_status",
            "parameter_identity_status",
            "solver_environment_status",
            "execution_policy",
            "uncertainty_hook_status",
        }
        if any(
            not isinstance(capability[field], str) or not capability[field]
            for field in scalar_capability_fields
        ):
            raise RouteArbitrationError(
                f"invalid capability summary for {route['route_id']}"
            )
        for field in ("action_modalities", "control_modes"):
            if not isinstance(capability[field], list) or not all(
                isinstance(value, str) and value for value in capability[field]
            ):
                raise RouteArbitrationError(
                    f"invalid capability list {field} for {route['route_id']}"
                )
        assumptions = route["assumptions"]
        if (
            not isinstance(assumptions, list)
            or not assumptions
            or not all(isinstance(value, str) and value for value in assumptions)
        ):
            raise RouteArbitrationError(
                f"route assumptions are required for {route['route_id']}"
            )
        criteria = route["criteria"]
        if not isinstance(criteria, list) or len(criteria) != len(RUBRIC_IDS):
            raise RouteArbitrationError(
                f"route must have exactly nine criteria: {route['route_id']}"
            )
        if [
            criterion.get("criterion_id")
            for criterion in criteria
            if isinstance(criterion, dict)
        ] != list(RUBRIC_IDS):
            raise RouteArbitrationError(
                f"route criterion identity or order drift: {route['route_id']}"
            )
        for criterion in criteria:
            _check_exact_keys(
                criterion,
                {"criterion_id", "result", "evidence_paths", "reason"},
                label="route criterion",
            )
            if criterion["result"] not in ALLOWED_RESULTS:
                raise RouteArbitrationError(
                    f"unknown criterion result for {route['route_id']}"
                )
            if not criterion["evidence_paths"] or len(
                criterion["evidence_paths"]
            ) != len(set(criterion["evidence_paths"])):
                raise RouteArbitrationError(
                    f"invalid criterion evidence paths for {route['route_id']}"
                )
            if not isinstance(criterion["reason"], str) or not criterion["reason"]:
                raise RouteArbitrationError(
                    f"missing criterion reason for {route['route_id']}"
                )
        if route["expected_disposition"] not in ALLOWED_DISPOSITIONS:
            raise RouteArbitrationError(
                f"invalid expected disposition for {route['route_id']}"
            )
        if not route["limitations"] or not route["prohibited_claims"]:
            raise RouteArbitrationError(
                f"route limitations and prohibited claims are required: {route['route_id']}"
            )

    route_by_id = {route["route_id"]: route for route in routes}
    if route_by_id["route_a"]["scientific_label"] != "synthetic_mechanistic_prior":
        raise RouteArbitrationError("Route A must retain synthetic_mechanistic_prior")
    if (
        route_by_id["route_b"]["availability_state"] != "not_available"
        or route_by_id["route_b"]["expected_disposition"] != "not_available"
    ):
        raise RouteArbitrationError(
            "Route B must remain not_available under parent authority"
        )
    if not EXPECTED_ABSENT_PATHS <= set(route_by_id["route_b"]["required_asset_paths"]):
        raise RouteArbitrationError(
            "Route B must bind both explicit absent checkpoint requirements"
        )
    if route_by_id["route_c"]["scientific_label"] != "prior_predictive":
        raise RouteArbitrationError("Route C must retain prior_predictive semantics")
    if (
        route_by_id["route_d"]["availability_state"] != "not_available"
        or route_by_id["route_d"]["expected_disposition"] != "not_available"
    ):
        raise RouteArbitrationError(
            "Route D must remain not_available while Route B is unavailable"
        )
    if route_by_id["route_d"]["depends_on_route_ids"] != ["route_a", "route_b"]:
        raise RouteArbitrationError(
            "Route D must depend on independently admissible Routes A and B"
        )
    if route_by_id["route_d"]["combination_rule_status"] != "absent_not_predeclared":
        raise RouteArbitrationError(
            "Route D must record the absent disagreement-preserving rule"
        )


def _read_asset_bytes(path: Path) -> bytes:
    return path.read_bytes()


def _run_git(
    repository: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["git", *args], cwd=repository, check=check, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RouteArbitrationError(
            f"Git identity query failed: git {' '.join(args)}"
        ) from exc


def collect_asset_git_provenance(
    repository_root: Path, assets: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    provenance: dict[str, dict[str, Any]] = {}
    for asset in assets:
        path = asset["path"]
        tracked = (
            _run_git(
                repository_root, "ls-files", "--error-unmatch", "--", path, check=False
            ).returncode
            == 0
        )
        status = _run_git(
            repository_root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            path,
        ).stdout.strip()
        if asset["expected_presence"] == "absent":
            if tracked or status:
                raise RouteArbitrationError(
                    f"expected-absent asset has Git identity: {path}"
                )
            provenance[path] = {"git_tracking_state": "absent", "git_blob_sha1": None}
            continue
        if not tracked or status:
            raise RouteArbitrationError(
                f"expected tracked-clean asset has Git drift: {path}: {status!r}"
            )
        fields = (
            _run_git(repository_root, "ls-files", "-s", "--", path)
            .stdout.strip()
            .split()
        )
        if len(fields) < 2:
            raise RouteArbitrationError(f"unable to resolve Git blob for {path}")
        provenance[path] = {
            "git_tracking_state": "tracked_clean",
            "git_blob_sha1": fields[1],
        }
    return provenance


def _validate_parent_payload(
    payload: Mapping[str, Any],
    validation_receipt: Mapping[str, Any],
    parent: Mapping[str, Any],
) -> None:
    if (
        validation_receipt.get("status") != "passed"
        or validation_receipt.get("artifact_count") != parent["artifact_count"]
    ):
        raise RouteArbitrationError(
            "existing parent-authority validator did not pass the frozen parent"
        )
    if validation_receipt.get("required_fact_count") != parent["required_fact_count"]:
        raise RouteArbitrationError("parent-authority required fact count drift")
    if (
        payload.get("schema_version") != parent["schema_version"]
        or payload.get("status") != parent["status"]
    ):
        raise RouteArbitrationError("parent-authority schema or status drift")
    if (
        payload.get("artifact_count") != parent["artifact_count"]
        or payload.get("blockers") != []
    ):
        raise RouteArbitrationError(
            "parent-authority artifact count or blocker state drift"
        )
    if payload.get("terminal_decisions") != TERMINAL_DECISIONS:
        raise RouteArbitrationError("parent negative decisions drift")
    if payload.get("next_action") != "arbitrate_no_task_fit_candidate_routes":
        raise RouteArbitrationError(
            "parent next_action no longer resolves to route arbitration"
        )
    if payload.get("aggregate_access") != PARENT_AGGREGATE_ACCESS:
        raise RouteArbitrationError(
            "parent authority contains missing or nonzero outcome/protected access"
        )
    if payload.get("authorization_closure") != PARENT_AUTHORIZATION_CLOSURE:
        raise RouteArbitrationError("parent authority reopens downstream authorization")


def _load_schema(repository_root: Path, relative: str) -> tuple[dict[str, Any], str]:
    path = _resolve_path(
        repository_root, relative, expected_presence="present", label="output schema"
    )
    raw = path.read_bytes()
    try:
        schema = json.loads(raw.decode("utf-8"))
        Draft202012Validator.check_schema(schema)
    except (UnicodeError, json.JSONDecodeError, SchemaError) as exc:
        raise RouteArbitrationError(
            f"invalid Draft 2020-12 schema {relative}: {exc}"
        ) from exc
    return _require_mapping(schema, label=relative), sha256_bytes(raw)


def validate_output_schema(
    payload: Mapping[str, Any], schema: Mapping[str, Any], *, label: str
) -> None:
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: list(error.path),
    )
    if errors:
        details = "; ".join(
            f"{'/'.join(str(part) for part in error.path) or '<root>'}: {error.message}"
            for error in errors
        )
        raise RouteArbitrationError(f"{label} schema validation failed: {details}")


def _reject_forbidden_output_keys(
    value: object, forbidden: set[str], path: tuple[str | int, ...] = ()
) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in forbidden:
                location = "/".join(str(part) for part in (*path, key))
                raise RouteArbitrationError(
                    f"forbidden performance or selection field in output: {location}"
                )
            _reject_forbidden_output_keys(item, forbidden, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_forbidden_output_keys(item, forbidden, (*path, index))


def _asset_identity(asset: Mapping[str, Any]) -> dict[str, Any]:
    return {"path": asset["path"], "sha256": asset["content_sha256"]}


def _build_asset_lineage(
    *,
    contract: Mapping[str, Any],
    contract_sha256: str,
    parent_payload: Mapping[str, Any],
    parent_validation: Mapping[str, Any],
    repository_root: Path,
    git_provenance: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    opened_present_paths: list[str] = []
    absent_paths: list[str] = []
    for asset in contract["assets"]:
        relative = asset["path"]
        expected_presence = asset["expected_presence"]
        path = _resolve_path(
            repository_root,
            relative,
            expected_presence=expected_presence,
            label="allowlisted asset",
        )
        observed_git = git_provenance.get(relative)
        if not isinstance(observed_git, Mapping):
            raise RouteArbitrationError(
                f"missing Git provenance for allowlisted asset: {relative}"
            )
        if (
            observed_git.get("git_tracking_state") != asset["git_tracking_state"]
            or observed_git.get("git_blob_sha1") != asset["git_blob_sha1"]
        ):
            raise RouteArbitrationError(
                f"Git identity drift for allowlisted asset: {relative}"
            )
        if expected_presence == "absent":
            absent_paths.append(relative)
            records.append(
                {
                    **asset,
                    "observed_presence": "absent",
                    "content_sha256": None,
                    "markers_verified": True,
                    "identity_matches_contract": True,
                }
            )
            records[-1].pop("sha256")
            continue

        raw = _read_asset_bytes(path)
        observed_hash = sha256_bytes(raw)
        if observed_hash != asset["sha256"] or len(raw) != asset["bytes"]:
            raise RouteArbitrationError(
                f"content identity drift for allowlisted asset: {relative}"
            )
        try:
            text = raw.decode("utf-8")
        except UnicodeError as exc:
            raise RouteArbitrationError(
                f"allowlisted text asset is not UTF-8: {relative}"
            ) from exc
        missing_markers = [
            marker for marker in asset["required_text_markers"] if marker not in text
        ]
        if missing_markers:
            raise RouteArbitrationError(
                f"required evidence markers missing from {relative}: {missing_markers}"
            )
        opened_present_paths.append(relative)
        records.append(
            {
                "asset_uid": asset["asset_uid"],
                "path": relative,
                "route_roles": list(asset["route_roles"]),
                "asset_kind": asset["asset_kind"],
                "provenance_role": asset["provenance_role"],
                "license_access_evidence_status": asset[
                    "license_access_evidence_status"
                ],
                "expected_presence": "present",
                "observed_presence": "present",
                "content_sha256": observed_hash,
                "bytes": len(raw),
                "git_blob_sha1": observed_git["git_blob_sha1"],
                "git_tracking_state": observed_git["git_tracking_state"],
                "classification": dict(asset["classification"]),
                "required_text_markers": list(asset["required_text_markers"]),
                "markers_verified": True,
                "identity_matches_contract": True,
                "limitations": list(asset["limitations"]),
            }
        )

    return {
        "schema_version": contract["outputs"]["reusable_asset_lineage"][
            "schema_version"
        ],
        "lineage_uid": "no_task_fit_reusable_asset_lineage",
        "status": contract["outputs"]["reusable_asset_lineage"]["status"],
        "contract": {"path": CONTRACT_PATH, "sha256": contract_sha256},
        "parent_authority": {
            "path": contract["parent_authority"]["path"],
            "sha256": contract["parent_authority"]["sha256"],
            "schema_version": parent_payload["schema_version"],
            "status": parent_payload["status"],
            "validation_status": parent_validation["status"],
        },
        "asset_count": len(records),
        "present_asset_count": len(opened_present_paths),
        "absent_asset_count": len(absent_paths),
        "assets": records,
        "access_boundary": dict(contract["access_boundary"]),
        "authorization_closure": dict(contract["authorization_closure"]),
        "source_path_scope": {
            "declared_path_count": len(records),
            "opened_present_paths": opened_present_paths,
            "verified_absent_paths": absent_paths,
            "non_allowlisted_paths_opened": [],
        },
        "limitations": [
            "Presence, byte identity, and Git identity do not imply route admissibility or scientific validity.",
            "No unlisted dependency, generated registry payload, checkpoint payload, or transitive artifact was inspected.",
            "License and access judgments are limited to declarations inside the frozen allowlisted assets.",
        ],
        "blockers": [],
    }


def _route_uid(
    route: Mapping[str, Any], asset_by_path: Mapping[str, Mapping[str, Any]]
) -> str:
    identity = {
        "route_id": route["route_id"],
        "scientific_label": route["scientific_label"],
        "availability_state": route["availability_state"],
        "required_assets": [
            {
                "path": path,
                "presence": asset_by_path[path]["observed_presence"],
                "sha256": asset_by_path[path]["content_sha256"],
            }
            for path in route["required_asset_paths"]
        ],
        "depends_on_route_ids": route["depends_on_route_ids"],
        "combination_rule_status": route["combination_rule_status"],
        "capability_support_summary": route["capability_support_summary"],
        "assumptions": route["assumptions"],
        "criteria": route["criteria"],
        "limitations": route["limitations"],
        "prohibited_claims": route["prohibited_claims"],
    }
    return f"no_task_fit_{route['route_id']}_{canonical_value_sha256(identity)[:16]}"


def _build_route_arbitration(
    *,
    contract: Mapping[str, Any],
    contract_sha256: str,
    parent_payload: Mapping[str, Any],
    asset_lineage: Mapping[str, Any],
    asset_lineage_sha256: str,
) -> dict[str, Any]:
    rubric_by_id = {row["criterion_id"]: row for row in contract["rubric"]}
    asset_by_path = {row["path"]: row for row in asset_lineage["assets"]}
    route_records: list[dict[str, Any]] = []
    dispositions: dict[str, str] = {}
    for route in contract["routes"]:
        missing_paths = [
            path
            for path in route["required_asset_paths"]
            if asset_by_path[path]["observed_presence"] == "absent"
        ]
        if route["availability_state"] == "available" and missing_paths:
            raise RouteArbitrationError(
                f"available route has missing required assets: {route['route_id']}"
            )
        criteria = [
            {
                "criterion_id": criterion["criterion_id"],
                "mandatory": rubric_by_id[criterion["criterion_id"]]["mandatory"],
                "result": criterion["result"],
                "evidence_paths": list(criterion["evidence_paths"]),
                "reason": criterion["reason"],
            }
            for criterion in route["criteria"]
        ]
        satisfied = sum(row["result"] == "satisfied" for row in criteria)
        unsatisfied = sum(row["result"] == "unsatisfied" for row in criteria)
        not_applicable = sum(row["result"] == "not_applicable" for row in criteria)
        if route["availability_state"] == "not_available":
            disposition = "not_available"
        elif unsatisfied == 0 and not_applicable == 0:
            disposition = "admissible"
        else:
            disposition = "inadmissible"
        if disposition != route["expected_disposition"]:
            raise RouteArbitrationError(
                f"computed disposition drift for {route['route_id']}"
            )
        if disposition == "admissible":
            disposition_reason = "Every mandatory structural criterion is satisfied."
        elif disposition == "inadmissible":
            disposition_reason = f"The route is available but {unsatisfied + not_applicable} mandatory structural criteria are not satisfied."
        else:
            disposition_reason = route["availability_reason"]
        route_record = {
            "route_id": route["route_id"],
            "route_uid": _route_uid(route, asset_by_path),
            "route_name": route["route_name"],
            "scientific_label": route["scientific_label"],
            "availability_state": route["availability_state"],
            "availability_reason": route["availability_reason"],
            "required_asset_paths": list(route["required_asset_paths"]),
            "missing_required_asset_paths": missing_paths,
            "depends_on_route_ids": list(route["depends_on_route_ids"]),
            "combination_rule_status": route["combination_rule_status"],
            "capability_support_summary": dict(route["capability_support_summary"]),
            "assumptions": list(route["assumptions"]),
            "criteria": criteria,
            "mandatory_criterion_count": len(criteria),
            "satisfied_mandatory_count": satisfied,
            "unsatisfied_mandatory_count": unsatisfied,
            "not_applicable_mandatory_count": not_applicable,
            "task_specific_fitting_performed": False,
            "performance_evidence_used": False,
            "limitations": list(route["limitations"]),
            "prohibited_claims": list(route["prohibited_claims"]),
            "disposition_reason": disposition_reason,
            "disposition": disposition,
        }
        route_records.append(route_record)
        dispositions[route["route_id"]] = disposition

    if dispositions["route_b"] != "not_available":
        raise RouteArbitrationError(
            "Route B must remain not_available under the parent checkpoint authority"
        )
    if (
        dispositions["route_d"] != "not_available"
        or dispositions["route_b"] == "admissible"
    ):
        raise RouteArbitrationError("Route D dependency rule was violated")
    admissible_route_ids = [
        row["route_id"] for row in route_records if row["disposition"] == "admissible"
    ]
    status = (
        "NO_TASK_FIT_ROUTES_ARBITRATED"
        if admissible_route_ids
        else "NO_ADMISSIBLE_NO_TASK_FIT_TWIN_ROUTE"
    )
    next_action = (
        "freeze_admissible_no_task_fit_candidates"
        if admissible_route_ids
        else "stop_no_admissible_no_task_fit_twin_route"
    )
    return {
        "schema_version": contract["outputs"]["route_arbitration"]["schema_version"],
        "arbitration_uid": "no_task_fit_candidate_route_arbitration",
        "status": status,
        "contract": {"path": CONTRACT_PATH, "sha256": contract_sha256},
        "parent_authority": {
            "path": contract["parent_authority"]["path"],
            "sha256": contract["parent_authority"]["sha256"],
            "status": parent_payload["status"],
        },
        "asset_lineage": {
            "path": LINEAGE_OUTPUT,
            "sha256": asset_lineage_sha256,
            "status": asset_lineage["status"],
            "asset_count": asset_lineage["asset_count"],
        },
        "rubric": [dict(row) for row in contract["rubric"]],
        "route_count": len(route_records),
        "routes": route_records,
        "admissible_route_ids": admissible_route_ids,
        "admissible_route_count": len(admissible_route_ids),
        "all_admissible_routes_retained": True,
        "candidate_selection_performed": False,
        "task_specific_fitting_performed": False,
        "performance_evidence_used": False,
        "access_boundary": dict(contract["access_boundary"]),
        "authorization_closure": dict(contract["authorization_closure"]),
        "decision_reasons": [
            f"{row['route_id']}:{row['disposition']}:{row['unsatisfied_mandatory_count']}_unsatisfied:{row['not_applicable_mandatory_count']}_not_applicable"
            for row in route_records
        ],
        "next_action": next_action,
    }


def project_candidate_registry(
    *,
    contract: Mapping[str, Any],
    contract_sha256: str,
    parent_payload: Mapping[str, Any],
    asset_lineage: Mapping[str, Any],
    asset_lineage_sha256: str,
    arbitration: Mapping[str, Any],
    arbitration_sha256: str,
) -> dict[str, Any]:
    asset_by_path = {row["path"]: row for row in asset_lineage["assets"]}
    admitted_routes = [
        row for row in arbitration["routes"] if row["disposition"] == "admissible"
    ]
    candidates: list[dict[str, Any]] = []
    for route in admitted_routes:
        identities: list[dict[str, str]] = []
        for path in route["required_asset_paths"]:
            asset = asset_by_path[path]
            if asset["observed_presence"] != "present" or not isinstance(
                asset["content_sha256"], str
            ):
                raise RouteArbitrationError(
                    f"admissible route depends on absent asset: {route['route_id']}:{path}"
                )
            identities.append(_asset_identity(asset))
        identity_basis = {
            "route_uid": route["route_uid"],
            "parent_authority_sha256": contract["parent_authority"]["sha256"],
            "asset_lineage_sha256": asset_lineage_sha256,
            "required_asset_identities": identities,
        }
        identity_hash = canonical_value_sha256(identity_basis)
        candidates.append(
            {
                "candidate_uid": f"no_task_fit_candidate_{route['route_id'][-1]}_{identity_hash[:16]}",
                "route_id": route["route_id"],
                "route_uid": route["route_uid"],
                "route_name": route["route_name"],
                "scientific_label": route["scientific_label"],
                "status": "frozen_admissible_candidate",
                "identity_basis_sha256": identity_hash,
                "parent_authority_sha256": contract["parent_authority"]["sha256"],
                "asset_lineage_sha256": asset_lineage_sha256,
                "required_asset_identities": identities,
                "task_specific_fitting_performed": False,
                "performance_evidence_used": False,
                "limitations": list(route["limitations"]),
            }
        )
    route_ids = [candidate["route_id"] for candidate in candidates]
    expected_route_ids = list(arbitration["admissible_route_ids"])
    if route_ids != expected_route_ids:
        raise RouteArbitrationError(
            "candidate registry is not an exact admissible-route projection"
        )
    status = (
        "NO_TASK_FIT_CANDIDATES_FROZEN"
        if candidates
        else "NO_ADMISSIBLE_NO_TASK_FIT_TWIN_ROUTE"
    )
    next_action = (
        "freeze_admissible_no_task_fit_candidates"
        if candidates
        else "stop_no_admissible_no_task_fit_twin_route"
    )
    return {
        "schema_version": contract["outputs"]["candidate_registry"]["schema_version"],
        "registry_uid": "no_task_fit_candidate_registry",
        "status": status,
        "contract": {"path": CONTRACT_PATH, "sha256": contract_sha256},
        "parent_authority": {
            "path": contract["parent_authority"]["path"],
            "sha256": contract["parent_authority"]["sha256"],
            "status": parent_payload["status"],
        },
        "asset_lineage": {
            "path": LINEAGE_OUTPUT,
            "sha256": asset_lineage_sha256,
            "status": asset_lineage["status"],
        },
        "route_arbitration": {
            "path": ARBITRATION_OUTPUT,
            "sha256": arbitration_sha256,
            "status": arbitration["status"],
        },
        "projection_rule": "all_and_only_admissible_routes",
        "admissible_route_ids": route_ids,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "candidate_selection_performed": False,
        "task_specific_fitting_performed": False,
        "performance_evidence_used": False,
        "access_boundary": dict(contract["access_boundary"]),
        "authorization_closure": dict(contract["authorization_closure"]),
        "limitations": [
            "The registry is a structural projection and contains no target-task performance result or preferred candidate.",
            "An empty registry is the valid terminal NO_ADMISSIBLE_NO_TASK_FIT_TWIN_ROUTE.",
        ],
        "next_action": next_action,
    }


def _validate_parent(
    repository_root: Path,
    contract: Mapping[str, Any],
    parent_validator: Callable[..., Mapping[str, Any]] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parent = contract["parent_authority"]
    parent_path = _resolve_path(
        repository_root,
        parent["path"],
        expected_presence="present",
        label="parent authority",
    )
    raw = parent_path.read_bytes()
    if sha256_bytes(raw) != parent["sha256"]:
        raise RouteArbitrationError("parent authority SHA-256 drift")
    try:
        parent_payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RouteArbitrationError(
            f"parent authority is not valid UTF-8 JSON: {exc}"
        ) from exc
    parent_payload = _require_mapping(parent_payload, label="parent authority")
    if parent_validator is None:
        from Libs.scripts.validators.validate_no_task_fit_parent_authority import (
            validate_bound_parent_authority,
        )

        validation = validate_bound_parent_authority(
            repository_root=repository_root,
            contract_path=parent["contract_path"],
            manifest_path=parent["path"],
        )
    else:
        validation = parent_validator(
            repository_root=repository_root,
            contract_path=parent["contract_path"],
            manifest_path=parent["path"],
        )
    validation = _require_mapping(validation, label="parent validation receipt")
    _validate_parent_payload(parent_payload, validation, parent)
    return parent_payload, validation


def compute_arbitration_outputs(
    *,
    repository_root: Path,
    contract_path: Path | str,
    parent_validator: Callable[..., Mapping[str, Any]] | None = None,
    asset_git_provenance: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Mapping[str, Any]]
]:
    repository_root = repository_root.resolve(strict=True)
    candidate_contract = Path(contract_path)
    if not candidate_contract.is_absolute():
        candidate_contract = repository_root / candidate_contract
    if candidate_contract.is_symlink():
        raise RouteArbitrationError("route-arbitration contract must not be a symlink")
    resolved_contract = candidate_contract.resolve(strict=True)
    if (
        not resolved_contract.is_relative_to(repository_root)
        or resolved_contract.relative_to(repository_root).as_posix() != CONTRACT_PATH
    ):
        raise RouteArbitrationError(
            f"route-arbitration contract must be {CONTRACT_PATH}"
        )
    contract, contract_raw = _load_contract(resolved_contract)
    _validate_contract(contract)
    contract_sha256 = sha256_bytes(contract_raw)

    parent_payload, parent_validation = _validate_parent(
        repository_root, contract, parent_validator
    )
    provenance = asset_git_provenance or collect_asset_git_provenance(
        repository_root, contract["assets"]
    )
    lineage = _build_asset_lineage(
        contract=contract,
        contract_sha256=contract_sha256,
        parent_payload=parent_payload,
        parent_validation=parent_validation,
        repository_root=repository_root,
        git_provenance=provenance,
    )
    lineage_bytes = canonical_json_bytes(lineage)
    lineage_sha256 = sha256_bytes(lineage_bytes)
    arbitration = _build_route_arbitration(
        contract=contract,
        contract_sha256=contract_sha256,
        parent_payload=parent_payload,
        asset_lineage=lineage,
        asset_lineage_sha256=lineage_sha256,
    )
    arbitration_bytes = canonical_json_bytes(arbitration)
    arbitration_sha256 = sha256_bytes(arbitration_bytes)
    registry = project_candidate_registry(
        contract=contract,
        contract_sha256=contract_sha256,
        parent_payload=parent_payload,
        asset_lineage=lineage,
        asset_lineage_sha256=lineage_sha256,
        arbitration=arbitration,
        arbitration_sha256=arbitration_sha256,
    )

    schemas: dict[str, Mapping[str, Any]] = {}
    for output_name, output in contract["outputs"].items():
        schema, _ = _load_schema(repository_root, output["schema_path"])
        schemas[output_name] = schema
    validate_output_schema(
        lineage, schemas["reusable_asset_lineage"], label="asset lineage"
    )
    validate_output_schema(
        arbitration, schemas["route_arbitration"], label="route arbitration"
    )
    validate_output_schema(
        registry, schemas["candidate_registry"], label="candidate registry"
    )
    for label, payload in (
        ("asset lineage", lineage),
        ("route arbitration", arbitration),
        ("candidate registry", registry),
    ):
        _reject_forbidden_output_keys(
            payload, set(contract["disposition_rules"]["forbidden_field_names"])
        )
    return lineage, arbitration, registry, schemas


def _install_outputs(repository_root: Path, outputs: Mapping[str, bytes]) -> None:
    output_root = repository_root / "Output" / "audits"
    output_root.mkdir(parents=True, exist_ok=True)
    for relative, raw in outputs.items():
        destination = (
            _resolve_path(
                repository_root,
                relative,
                expected_presence="absent",
                label="new arbitration output",
            )
            if not (repository_root / relative).exists()
            else repository_root / relative
        )
        if destination.is_symlink() or (
            destination.exists() and not destination.is_file()
        ):
            raise RouteArbitrationError(
                f"output destination is not a regular non-symlink path: {relative}"
            )
        if destination.exists() and destination.read_bytes() != raw:
            raise RouteArbitrationError(
                f"refusing to overwrite drifted route-arbitration output: {relative}"
            )
    created: list[Path] = []
    with tempfile.TemporaryDirectory(
        prefix=".no_task_fit_route_arbitration_", dir=output_root
    ) as temporary:
        staged_root = Path(temporary)
        for relative, raw in outputs.items():
            staged = staged_root / relative
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_bytes(raw)
            if staged.read_bytes() != raw:
                raise RouteArbitrationError(f"staged output drift: {relative}")
        try:
            for relative, raw in outputs.items():
                destination = repository_root / relative
                if destination.exists():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged_root / relative, destination)
                created.append(destination)
                if destination.read_bytes() != raw:
                    raise RouteArbitrationError(f"installed output drift: {relative}")
        except Exception as exc:
            for destination in reversed(created):
                try:
                    destination.unlink()
                except FileNotFoundError:
                    pass
            if isinstance(exc, RouteArbitrationError):
                raise
            raise RouteArbitrationError(
                f"transactional route output installation failed: {exc}"
            ) from exc


def build_route_arbitration(
    *,
    repository_root: Path,
    contract_path: Path | str,
    parent_validator: Callable[..., Mapping[str, Any]] | None = None,
    asset_git_provenance: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    lineage, arbitration, registry, _ = compute_arbitration_outputs(
        repository_root=repository_root,
        contract_path=contract_path,
        parent_validator=parent_validator,
        asset_git_provenance=asset_git_provenance,
    )
    outputs = {
        LINEAGE_OUTPUT: canonical_json_bytes(lineage),
        ARBITRATION_OUTPUT: canonical_json_bytes(arbitration),
        CANDIDATE_OUTPUT: canonical_json_bytes(registry),
    }
    _install_outputs(repository_root.resolve(strict=True), outputs)
    return lineage, arbitration, registry


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        lineage, arbitration, registry = build_route_arbitration(
            repository_root=args.repository_root,
            contract_path=args.contract,
        )
    except RouteArbitrationError as exc:
        raise SystemExit(f"no-task-fit route arbitration failed: {exc}") from exc
    print(
        json.dumps(
            {
                "asset_count": lineage["asset_count"],
                "present_asset_count": lineage["present_asset_count"],
                "absent_asset_count": lineage["absent_asset_count"],
                "route_dispositions": {
                    row["route_id"]: row["disposition"] for row in arbitration["routes"]
                },
                "status": arbitration["status"],
                "candidate_count": registry["candidate_count"],
                "next_action": registry["next_action"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
