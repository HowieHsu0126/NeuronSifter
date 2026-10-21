"""Independently validate all controlled neural world packages and lineage."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Mapping

from jsonschema import Draft202012Validator


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.action_observation_contracts import validate_world_action  # noqa: E402
from Libs.utils.data.controlled_world_contracts import (  # noqa: E402
    AUDIT_SCHEMA_VERSION,
    CLAIM_BOUNDARY,
    FAMILY_UIDS,
    ControlledWorldContractError,
    bind_information_action,
    classify_world_action,
    ensure_no_runtime_secrets,
    select_family_config,
    validate_controlled_world_config,
    validate_family_manifest,
    validate_replay_manifest,
    validate_runtime_instance,
    validate_zero_access,
)
from Libs.utils.data.controlled_world_runtime import (  # noqa: E402
    generate_runtime_instance,
    replay_runtime_instance,
    sha256_file,
)
from Libs.utils.data.raw_data_registry import load_config, write_json  # noqa: E402
from Libs.utils.data.trajectory_contracts import canonical_sha256, sorted_blockers  # noqa: E402


WORLD_SCHEMA_PATH = Path("Libs/configs/schema/controlled_neural_world.schema.json")
REPLAY_SCHEMA_PATH = Path("Libs/configs/schema/controlled_world_replay_manifest.schema.json")
CONTROLLED_CONFIG_PATH = Path("Libs/configs/data/controlled_neural_world.yaml")
VALIDATOR_UID = "controlled_neural_world_contract_validator"


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _schema_errors(schema_path: Path, value: Mapping[str, object], artifact: str) -> list[dict[str, object]]:
    schema = _load_json(schema_path)
    Draft202012Validator.check_schema(schema)
    return [
        {
            "code": "schema_validation_failed",
            "message": error.message,
            "artifact": artifact,
            "path": ".".join(str(part) for part in error.absolute_path),
        }
        for error in sorted(Draft202012Validator(schema).iter_errors(dict(value)), key=lambda item: list(item.absolute_path))
    ]


def _resolve_artifact(artifact_root: Path, relative_path: object) -> Path:
    relative = Path(str(relative_path))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Controlled-world output path must be repository-relative: {relative}")
    return artifact_root / relative


def _blocker_from_exception(error: Exception, *, artifact: str) -> dict[str, object]:
    if isinstance(error, ControlledWorldContractError):
        return {**error.to_blocker(), "artifact": artifact}
    return {"code": "validation_exception", "message": str(error), "artifact": artifact}


def _action_sensitivity_check(
    source: Mapping[str, object],
    family_uid: str,
    *,
    repository_root: Path,
    config_file_identity: str,
) -> bool:
    family = select_family_config(source, family_uid)
    action = deepcopy(family["world_action"]["canonical"])
    extension = deepcopy(family["world_action"]["extension"])
    bounds = family["support_domain"]["supported_intensity"]
    alternative = float(bounds["minimum"]) + 0.2 * (float(bounds["maximum"]) - float(bounds["minimum"]))
    if alternative == float(extension["dose_or_intensity"]["value"]):
        alternative = float(bounds["minimum"]) + 0.8 * (float(bounds["maximum"]) - float(bounds["minimum"]))
    extension["dose_or_intensity"]["value"] = alternative
    if action["intervention_type"] == "pharmacological":
        action["dose"] = alternative
    baseline, _ = generate_runtime_instance(
        source,
        family_uid=family_uid,
        repository_root=repository_root,
        config_file_identity=config_file_identity,
    )
    altered, _ = generate_runtime_instance(
        source,
        family_uid=family_uid,
        repository_root=repository_root,
        config_file_identity=config_file_identity,
        canonical_action=action,
        action_extension=extension,
    )
    delay = int(family["world_action"]["response_delay_steps"])
    if baseline["observations"][:delay] != altered["observations"][:delay]:
        return False
    return any(
        baseline["observations"][index] != altered["observations"][index]
        for index in range(delay, len(baseline["observations"]))
    )


def validate_controlled_neural_world(
    *,
    config_path: Path,
    repository_root: Path,
    artifact_root: Path,
    write_audit: bool = True,
) -> dict[str, object]:
    blockers: list[dict[str, object]] = []
    checks = {
        "config_schema": False,
        "exact_family_set": False,
        "runtime_schemas": False,
        "replay_schemas": False,
        "family_manifest_schema": False,
        "oracle_firewall": False,
        "canonical_actions": False,
        "information_partial_failure": False,
        "support_recomputed": False,
        "family_causal_invariants": False,
        "lineage_complete": False,
        "deterministic_replay": False,
        "aggregate_content_addresses": False,
        "cross_family_uniqueness": False,
        "zero_protected_access": False,
        "non_qualification_boundary": False,
    }
    family_results: list[dict[str, object]] = []
    loaded_packages: dict[str, tuple[dict[str, object], dict[str, object]]] = {}
    family_manifest: dict[str, object] = {}
    source: dict[str, object] = {}
    try:
        config = load_config(config_path)
        source = validate_controlled_world_config(config)
        errors = _schema_errors(repository_root / WORLD_SCHEMA_PATH, source, "config")
        blockers.extend(errors)
        checks["config_schema"] = not errors
        checks["exact_family_set"] = [row["family_uid"] for row in source["families"]] == list(FAMILY_UIDS)
        if not checks["exact_family_set"]:
            blockers.append({"code": "family_order_invalid", "message": "Source config family order is not canonical.", "artifact": "config"})
    except Exception as error:
        blockers.append(_blocker_from_exception(error, artifact="config"))
        config = {}

    config_file_identity = sha256_file(repository_root / CONTROLLED_CONFIG_PATH) if source else ""
    runtime_schema_ok = True
    replay_schema_ok = True
    oracle_ok = True
    action_ok = True
    information_ok = True
    support_ok = True
    causal_ok = True
    lineage_ok = True
    replay_ok = True
    zero_access_ok = True
    if source:
        for family_uid in FAMILY_UIDS:
            family = select_family_config(source, family_uid)
            instance_path = _resolve_artifact(artifact_root, family["output_paths"]["world_instance"])
            replay_path = _resolve_artifact(artifact_root, family["output_paths"]["replay_manifest"])
            family_checks = {
                "runtime_schema": False,
                "replay_schema": False,
                "oracle_firewall": False,
                "canonical_action": False,
                "information_partial_failure": False,
                "support_recomputed": False,
                "causal_invariants": False,
                "lineage_complete": False,
                "deterministic_replay": False,
                "zero_protected_access": False,
            }
            instance: dict[str, object] = {}
            replay: dict[str, object] = {}
            artifact_prefix = family_uid
            if not instance_path.is_file():
                blockers.append({"code": "artifact_missing", "message": "Family runtime artifact is missing.", "artifact": f"{artifact_prefix}.world_instance"})
                runtime_schema_ok = oracle_ok = action_ok = information_ok = support_ok = causal_ok = zero_access_ok = False
            else:
                try:
                    instance = _load_json(instance_path)
                    errors = _schema_errors(repository_root / WORLD_SCHEMA_PATH, instance, f"{artifact_prefix}.world_instance")
                    blockers.extend(errors)
                    family_checks["runtime_schema"] = not errors
                    runtime_schema_ok &= not errors
                    validate_runtime_instance(
                        instance,
                        config=source,
                        repository_root=repository_root,
                        config_file_identity=config_file_identity,
                    )
                    ensure_no_runtime_secrets(instance)
                    family_checks["oracle_firewall"] = True
                    binding = dict(instance["action_binding"])
                    action = dict(binding["canonical_world_action"])
                    if validate_world_action(action):
                        raise ControlledWorldContractError("invalid_world_action", "Runtime WorldAction validation failed.")
                    family_checks["canonical_action"] = True
                    support = classify_world_action(family, action, dict(binding["controlled_extension"]))
                    if support.to_dict() != instance["support_status"] or support.status != "supported":
                        raise ControlledWorldContractError("support_recomputation_mismatch", "Stored support status differs from recomputation.")
                    family_checks["support_recomputed"] = True
                    expected_bindings = [bind_information_action(family, dict(spec)).to_dict() for spec in family["information_actions"]]
                    results = list(instance["information_results"])
                    statuses = {str(row["status"]) for row in results}
                    if len(results) != len(expected_bindings) or statuses != {"partial", "failed"} or not all(row["missing_observables"] for row in results):
                        raise ControlledWorldContractError("information_semantics_invalid", "Partial/failure information semantics are incomplete.")
                    family_checks["information_partial_failure"] = True
                    if not _action_sensitivity_check(
                        source,
                        family_uid,
                        repository_root=repository_root,
                        config_file_identity=config_file_identity,
                    ):
                        raise ControlledWorldContractError("family_causal_invariant_failed", "Delayed family action sensitivity was not reproduced.")
                    family_checks["causal_invariants"] = True
                    validate_zero_access(dict(instance["access_counters"]))
                    family_checks["zero_protected_access"] = True
                except Exception as error:
                    blockers.append(_blocker_from_exception(error, artifact=f"{artifact_prefix}.world_instance"))
                    oracle_ok &= family_checks["oracle_firewall"]
                    action_ok &= family_checks["canonical_action"]
                    information_ok &= family_checks["information_partial_failure"]
                    support_ok &= family_checks["support_recomputed"]
                    causal_ok &= family_checks["causal_invariants"]
                    zero_access_ok &= family_checks["zero_protected_access"]
            if not replay_path.is_file():
                blockers.append({"code": "artifact_missing", "message": "Family replay artifact is missing.", "artifact": f"{artifact_prefix}.replay_manifest"})
                replay_schema_ok = lineage_ok = replay_ok = zero_access_ok = False
            else:
                try:
                    replay = _load_json(replay_path)
                    errors = _schema_errors(repository_root / REPLAY_SCHEMA_PATH, replay, f"{artifact_prefix}.replay_manifest")
                    blockers.extend(errors)
                    family_checks["replay_schema"] = not errors
                    replay_schema_ok &= not errors
                    validate_replay_manifest(replay)
                    required_identities = (
                        "generator_code_identity", "config_content_identity", "config_file_identity", "seed_identity",
                        "action_sequence_identity", "information_sequence_identity", "observable_output_identity",
                    )
                    if not all(isinstance(replay.get(field), str) and len(str(replay[field])) == 64 for field in required_identities):
                        raise ControlledWorldContractError("lineage_incomplete", "Replay identity lineage is incomplete.")
                    family_checks["lineage_complete"] = True
                    validate_zero_access(dict(replay["access_counters"]))
                    family_checks["zero_protected_access"] = family_checks["zero_protected_access"] and True
                    replayed = replay_runtime_instance(source, replay, repository_root=repository_root)
                    if not instance or canonical_sha256(replayed) != canonical_sha256(instance) or canonical_sha256(instance) != replay["observable_output_identity"]:
                        raise ControlledWorldContractError("replay_output_mismatch", "Replay did not reproduce the stored output.")
                    family_checks["deterministic_replay"] = True
                except Exception as error:
                    blockers.append(_blocker_from_exception(error, artifact=f"{artifact_prefix}.replay_manifest"))
                    lineage_ok &= family_checks["lineage_complete"]
                    replay_ok &= family_checks["deterministic_replay"]
                    zero_access_ok &= family_checks["zero_protected_access"]
            oracle_ok &= family_checks["oracle_firewall"]
            action_ok &= family_checks["canonical_action"]
            information_ok &= family_checks["information_partial_failure"]
            support_ok &= family_checks["support_recomputed"]
            causal_ok &= family_checks["causal_invariants"]
            lineage_ok &= family_checks["lineage_complete"]
            replay_ok &= family_checks["deterministic_replay"]
            zero_access_ok &= family_checks["zero_protected_access"]
            if instance and replay:
                loaded_packages[family_uid] = (instance, replay)
            family_results.append(
                {
                    "family_uid": family_uid,
                    "world_uid": dict(instance.get("world_identity", {})).get("world_uid"),
                    "observable_output_identity": replay.get("observable_output_identity"),
                    "status": "passed" if all(family_checks.values()) else "blocked",
                    "checks": family_checks,
                }
            )

        checks["runtime_schemas"] = runtime_schema_ok
        checks["replay_schemas"] = replay_schema_ok
        checks["oracle_firewall"] = oracle_ok
        checks["canonical_actions"] = action_ok
        checks["information_partial_failure"] = information_ok
        checks["support_recomputed"] = support_ok
        checks["family_causal_invariants"] = causal_ok
        checks["lineage_complete"] = lineage_ok
        checks["deterministic_replay"] = replay_ok
        checks["zero_protected_access"] = zero_access_ok
        manifest_path = _resolve_artifact(artifact_root, source["family_manifest"]["output_path"])
        if not manifest_path.is_file():
            blockers.append({"code": "artifact_missing", "message": "Aggregate family manifest is missing.", "artifact": "family_manifest"})
        else:
            try:
                family_manifest = _load_json(manifest_path)
                errors = _schema_errors(repository_root / WORLD_SCHEMA_PATH, family_manifest, "family_manifest")
                blockers.extend(errors)
                checks["family_manifest_schema"] = not errors
                validate_family_manifest(
                    family_manifest,
                    config=source,
                    config_file_identity=config_file_identity,
                    authoritative_packages=loaded_packages,
                )
                manifest_rows = {row["family_uid"]: row for row in family_manifest["families"]}
                for family_uid, (instance, replay) in loaded_packages.items():
                    row = manifest_rows[family_uid]
                    if row["runtime_content_identity"] != canonical_sha256(instance) or row["replay_content_identity"] != canonical_sha256(replay):
                        raise ControlledWorldContractError("aggregate_hash_mismatch", "Aggregate family content address differs from stored artifacts.", family_uid=family_uid)
                checks["aggregate_content_addresses"] = len(loaded_packages) == 3
                checks["cross_family_uniqueness"] = True
                checks["non_qualification_boundary"] = (
                    family_manifest["benchmark_split_assignment"] == "not_created"
                    and family_manifest["sealed_test_partitions"] == "not_created"
                    and family_manifest["information_headroom_qualification"] == "not_run"
                    and family_manifest["benchmark_qualification"] == "not_issued"
                    and family_manifest["claim_boundary"] == CLAIM_BOUNDARY
                )
                if not checks["non_qualification_boundary"]:
                    raise ControlledWorldContractError("benchmark_scope_violation", "Aggregate manifest oversteps implementation scope.")
            except Exception as error:
                blockers.append(_blocker_from_exception(error, artifact="family_manifest"))

    blockers = sorted_blockers(blockers)
    audit = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "validator_uid": VALIDATOR_UID,
        "status": "passed" if not blockers and all(checks.values()) else "blocked",
        "blocker_count": len(blockers),
        "blockers": blockers,
        "checks": checks,
        "family_count": len(family_results),
        "family_results": family_results,
        "family_manifest_identity": canonical_sha256(family_manifest) if family_manifest else None,
        "access_counters": {"six_context_outcome_access": 0, "protected_holdout_access": 0},
        "claim_boundary": dict(source.get("claim_boundary", CLAIM_BOUNDARY)),
    }
    audit_errors = _schema_errors(repository_root / WORLD_SCHEMA_PATH, audit, "validation_audit")
    if audit_errors:
        audit["status"] = "blocked"
        audit["blockers"] = sorted_blockers([*blockers, *audit_errors])
        audit["blocker_count"] = len(audit["blockers"])
    audit_path = _resolve_artifact(
        artifact_root,
        source.get("validation_audit_path", "Output/audits/controlled_neural_world_contract.json"),
    )
    if write_audit:
        write_json(audit_path, audit)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "Libs/configs/config_stack.yaml")
    parser.add_argument("--repository-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    audit = validate_controlled_neural_world(
        config_path=args.config.resolve(),
        repository_root=args.repository_root.resolve(),
        artifact_root=args.artifact_root.resolve(),
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
