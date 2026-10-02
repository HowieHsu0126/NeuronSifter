"""Read-only validation for the frozen-Twin static decision loop."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.models.modules.static_information_policy import POLICY_UIDS  # noqa: E402
from Libs.utils.data.public_foundation_corpus import sha256_file  # noqa: E402


DEFAULT_CONTRACT = (
    REPO_ROOT / "Libs/configs/neuronsifter/static_decision_loop.yaml"
)
TERMINAL_STATES = {
    "neuronsifter_static_decision_loop_development_resolved",
    "partial_static_decision_loop_evidence",
    "failed_static_decision_loop",
    "blocked_static_decision_loop",
}
ROUND_RECEIPT_SCHEMA_VERSION = "neuronsifter-static-loop-round-receipt-v1"
RELIABILITY_PROXY_NAME = "calibrated_selective_reliability_proxy_v1"


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Contract must be a mapping: {path}")
    return value


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _parse_utc_timestamp(value: Any, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid {field} timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp lacks timezone for {field}: {value!r}")
    return parsed.astimezone(timezone.utc)


def validate_evidence_boundary(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    blockers: list[str] = []
    for index, row in enumerate(rows):
        if row.get("evidence_role") != "development_only":
            blockers.append(f"row_{index}:evidence_role")
        if _truthy(row.get("publication_claim_eligible")):
            blockers.append(f"row_{index}:publication_claim_eligible")
        try:
            protected = int(row.get("protected_holdout_access", -1))
        except (TypeError, ValueError):
            protected = -1
        if protected != 0:
            blockers.append(f"row_{index}:protected_holdout_access")
    return blockers


def validate_posterior_chain(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_initial_posterior_uid: str,
    expected_rounds: int,
) -> list[str]:
    blockers: list[str] = []
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("policy_uid", "")), []).append(row)
    if set(grouped) != set(POLICY_UIDS):
        blockers.append("policy_chain_membership")
    for policy_uid in POLICY_UIDS:
        policy_rows = sorted(
            grouped.get(policy_uid, []), key=lambda row: str(row.get("round_uid", ""))
        )
        if len(policy_rows) != expected_rounds:
            blockers.append(f"{policy_uid}:round_count")
            continue
        previous_post = expected_initial_posterior_uid
        expected_version: int | None = None
        seen_results: set[str] = set()
        for round_index, row in enumerate(policy_rows, start=1):
            prefix = f"{policy_uid}:round_{round_index}"
            if row.get("round_uid") != f"round_{round_index}":
                blockers.append(f"{prefix}:round_identity")
            if row.get("posterior_uid_pre") != previous_post:
                blockers.append(f"{prefix}:mixed_chain_parent")
            if not row.get("world_action_uid") or not row.get("information_action_uid"):
                blockers.append(f"{prefix}:typed_action_identity")
            result_uid = str(row.get("result_uid", ""))
            if not result_uid or result_uid in seen_results:
                blockers.append(f"{prefix}:result_identity")
            seen_results.add(result_uid)
            if not _truthy(row.get("model_checkpoint_pre_equals_post")):
                blockers.append(f"{prefix}:checkpoint_changed")
            if _truthy(row.get("twin_parameters_changed")):
                blockers.append(f"{prefix}:twin_parameters_changed")
            if _truthy(row.get("agent_parameters_changed")):
                blockers.append(f"{prefix}:agent_parameters_changed")
            if not _truthy(row.get("parent_link_valid")):
                blockers.append(f"{prefix}:parent_link")
            if not _truthy(row.get("shared_posterior_invariant")):
                blockers.append(f"{prefix}:posterior_invariant")
            admitted = _truthy(row.get("observation_admitted"))
            pre_uid = str(row.get("posterior_uid_pre", ""))
            post_uid = str(row.get("posterior_uid_post", ""))
            try:
                pre_version = int(row.get("posterior_version_pre", 0))
                post_version = int(row.get("posterior_version_post", 0))
            except (TypeError, ValueError):
                blockers.append(f"{prefix}:posterior_version_parse")
                pre_version = post_version = 0
            if expected_version is None:
                expected_version = pre_version
            if pre_version != expected_version:
                blockers.append(f"{prefix}:posterior_version_pre")
            if admitted:
                if post_uid == pre_uid or post_version != pre_version + 1:
                    blockers.append(f"{prefix}:tampered_transition")
            elif post_uid != pre_uid or post_version != pre_version:
                blockers.append(f"{prefix}:failed_result_moved_state")
            previous_post = post_uid
            expected_version = post_version
    return blockers


def _require_files(paths: Mapping[str, Path]) -> list[str]:
    return [f"missing:{name}:{path}" for name, path in paths.items() if not path.is_file()]


def validate_recovery_catalog(
    recovery: Mapping[str, Any],
    *,
    outputs: Mapping[str, Path],
    expected_rounds: int,
) -> list[str]:
    """Revalidate the recovery catalog's embedded artifact identities.

    The catalog is a derived audit, so its status fields are not sufficient on
    their own. Every receipt and bound/derived artifact must still resolve to
    the contract-declared path and match the recorded content hash.
    """

    blockers: list[str] = []
    run_root = outputs["run_root"]
    receipt_rows = recovery.get("round_receipts")
    if not isinstance(receipt_rows, list):
        return ["recovery_round_receipts_not_list"]
    expected_keys = {
        (policy_uid, f"round_{round_index}")
        for policy_uid in POLICY_UIDS
        for round_index in range(1, expected_rounds + 1)
    }
    observed_keys: set[tuple[str, str]] = set()
    if len(receipt_rows) != len(expected_keys):
        blockers.append("recovery_receipt_row_count")
    for index, row in enumerate(receipt_rows):
        if not isinstance(row, Mapping):
            blockers.append(f"recovery_receipt_{index}:not_mapping")
            continue
        policy_uid = str(row.get("policy_uid", ""))
        round_uid = str(row.get("round_uid", ""))
        key = (policy_uid, round_uid)
        if key in observed_keys:
            blockers.append(f"recovery_receipt_{index}:duplicate_round")
        observed_keys.add(key)
        expected_path = run_root / policy_uid / round_uid / "round_receipt.json"
        receipt_path = _repo_path(str(row.get("receipt_ref", "")))
        if receipt_path.resolve() != expected_path.resolve():
            blockers.append(f"recovery_receipt_{index}:path")
            continue
        if not receipt_path.is_file():
            blockers.append(f"recovery_receipt_{index}:missing")
            continue
        if sha256_file(receipt_path) != str(row.get("receipt_sha256", "")):
            blockers.append(f"recovery_receipt_{index}:hash")
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            blockers.append(f"recovery_receipt_{index}:parse:{exc}")
            continue
        for field in (
            "policy_uid",
            "round_uid",
            "receipt_uid",
            "freeze_before_dispatch_verified",
            "controller_timestamp_source",
        ):
            if receipt.get(field) != row.get(field):
                blockers.append(f"recovery_receipt_{index}:{field}")
    if observed_keys != expected_keys:
        blockers.append("recovery_receipt_round_membership")

    expected_bound = {
        "posterior_lineage": outputs["posterior_lineage"],
        "tool_catalog": outputs["tool_catalog"],
        "decision_loop": outputs["decision_loop"],
        "architecture_gate": outputs["architecture_gate"],
    }
    expected_derived = {
        "prospective_manifest": outputs["prospective_manifest"],
        "information_effects": outputs["information_effects"],
    }
    for group_name, expected in (
        ("bound_read_only_artifacts", expected_bound),
        ("derived_artifacts_updated", expected_derived),
    ):
        bindings = recovery.get(group_name)
        if not isinstance(bindings, Mapping):
            blockers.append(f"recovery_{group_name}_not_mapping")
            continue
        if set(bindings) != set(expected):
            blockers.append(f"recovery_{group_name}_membership")
        for name, expected_path in expected.items():
            binding = bindings.get(name)
            if not isinstance(binding, Mapping):
                blockers.append(f"recovery_{group_name}:{name}:not_mapping")
                continue
            actual_path = _repo_path(str(binding.get("path", "")))
            if actual_path.resolve() != expected_path.resolve():
                blockers.append(f"recovery_{group_name}:{name}:path")
                continue
            if not actual_path.is_file():
                blockers.append(f"recovery_{group_name}:{name}:missing")
                continue
            if sha256_file(actual_path) != str(binding.get("sha256", "")):
                blockers.append(f"recovery_{group_name}:{name}:hash")
    return blockers


def validate_round_artifacts(
    *,
    run_root: Path,
    prospective_rows: Sequence[Mapping[str, Any]],
    lineage_rows: Sequence[Mapping[str, Any]],
    expected_rounds: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    blockers: list[str] = []
    summaries: list[dict[str, Any]] = []
    prospective_by_key = {
        (str(row.get("policy_uid", "")), str(row.get("round_uid", ""))): row
        for row in prospective_rows
    }
    lineage_by_key = {
        (str(row.get("policy_uid", "")), str(row.get("round_uid", ""))): row
        for row in lineage_rows
    }
    if len(prospective_by_key) != len(prospective_rows):
        blockers.append("prospective_duplicate_policy_round")
    if len(lineage_by_key) != len(lineage_rows):
        blockers.append("lineage_duplicate_policy_round")
    for policy_uid in POLICY_UIDS:
        for round_index in range(1, expected_rounds + 1):
            round_uid = f"round_{round_index}"
            prefix = f"{policy_uid}:{round_uid}"
            round_root = run_root / policy_uid / round_uid
            paths = {
                "input_manifest": round_root / f"{round_uid}_input_manifest.json",
                "selection": round_root
                / f"{round_uid}_selected_information_actions.tsv",
                "tool_results": round_root / f"{round_uid}_tool_results.tsv",
                "evidence_admission": round_root
                / f"{round_uid}_evidence_admission.tsv",
                "posterior_update": round_root / f"{round_uid}_posterior_update.json",
                "scientific_pre_post": round_root
                / f"{round_uid}_scientific_pre_post.tsv",
                "report": round_root / f"{round_uid}_report.json",
                "receipt": round_root / "round_receipt.json",
            }
            missing = _require_files(paths)
            if missing:
                blockers.extend(f"{prefix}:{item}" for item in missing)
                continue
            try:
                manifest = json.loads(
                    paths["input_manifest"].read_text(encoding="utf-8")
                )
                selection_rows = _read_tsv(paths["selection"])
                receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                blockers.append(f"{prefix}:round_artifact_parse:{exc}")
                continue
            if len(selection_rows) != 1:
                blockers.append(f"{prefix}:selection_cardinality")
                continue
            selection = selection_rows[0]
            lineage = lineage_by_key.get((policy_uid, round_uid))
            prospective = prospective_by_key.get((policy_uid, round_uid))
            if lineage is None:
                blockers.append(f"{prefix}:lineage_missing")
                continue
            if prospective is None:
                blockers.append(f"{prefix}:prospective_missing")
                continue
            if manifest.get("outcome_visible_before_freeze") is not False:
                blockers.append(f"{prefix}:outcome_visible_before_freeze")
            if not _truthy(selection.get("selection_before_outcome")):
                blockers.append(f"{prefix}:selection_before_outcome")
            if receipt.get("schema_version") != ROUND_RECEIPT_SCHEMA_VERSION:
                blockers.append(f"{prefix}:receipt_schema")
            receipt_payload = {
                key: value
                for key, value in receipt.items()
                if key not in {"receipt_payload_sha256", "receipt_uid"}
            }
            expected_payload_hash = _digest(receipt_payload)
            if receipt.get("receipt_payload_sha256") != expected_payload_hash:
                blockers.append(f"{prefix}:receipt_payload_hash")
            if receipt.get("receipt_uid") != (
                f"static-loop-round-receipt-{expected_payload_hash[:32]}"
            ):
                blockers.append(f"{prefix}:receipt_uid")
            if receipt.get("outcome_visible_before_freeze") is not False:
                blockers.append(f"{prefix}:receipt_outcome_visibility")
            if receipt.get("selection_before_outcome") is not True:
                blockers.append(f"{prefix}:receipt_selection_order")
            if receipt.get("freeze_before_dispatch_verified") is not True:
                blockers.append(f"{prefix}:receipt_freeze_order")
            artifact_bindings = receipt.get("artifact_bindings", {})
            for name, artifact_path in paths.items():
                if name in {"receipt", "selection"}:
                    binding_name = (
                        "selected_information_action" if name == "selection" else name
                    )
                else:
                    binding_name = name
                if name == "receipt":
                    continue
                binding = artifact_bindings.get(binding_name, {})
                if _repo_path(binding.get("path", "")) != artifact_path.resolve():
                    blockers.append(f"{prefix}:{binding_name}_path")
                    continue
                if sha256_file(artifact_path) != binding.get("sha256"):
                    blockers.append(f"{prefix}:{binding_name}_hash")
                stat = artifact_path.stat()
                if int(binding.get("size_bytes", -1)) != int(stat.st_size):
                    blockers.append(f"{prefix}:{binding_name}_size")
                if int(binding.get("mtime_ns", -1)) != int(stat.st_mtime_ns):
                    blockers.append(f"{prefix}:{binding_name}_mtime")
            dispatch_bindings = receipt.get("dispatch_bindings", [])
            if {row.get("action_role") for row in dispatch_bindings} != {
                "intervention",
                "matched_control",
            }:
                blockers.append(f"{prefix}:dispatch_roles")
            dispatch_starts: list[datetime] = []
            for dispatch in dispatch_bindings:
                dispatch_path = _repo_path(dispatch.get("dispatch_ref", ""))
                if not dispatch_path.is_file():
                    blockers.append(f"{prefix}:dispatch_missing")
                    continue
                if sha256_file(dispatch_path) != dispatch.get("dispatch_sha256"):
                    blockers.append(f"{prefix}:dispatch_hash")
                raw = json.loads(dispatch_path.read_text(encoding="utf-8"))
                for field in ("raw_bytes_hash", "normalized_bytes_hash", "status", "qc"):
                    if str(raw.get(field, "")) != str(dispatch.get(field, "")):
                        blockers.append(
                            f"{prefix}:dispatch_{dispatch.get('action_role')}:{field}"
                        )
                try:
                    dispatch_starts.append(
                        _parse_utc_timestamp(
                            dispatch.get("started_at_controller"),
                            field=f"{prefix}.started_at_controller",
                        )
                    )
                except ValueError:
                    blockers.append(f"{prefix}:dispatch_started_at_controller")
            freeze_mtime_ns = max(
                paths["input_manifest"].stat().st_mtime_ns,
                paths["selection"].stat().st_mtime_ns,
            )
            if int(receipt.get("freeze_latest_mtime_ns", -1)) != freeze_mtime_ns:
                blockers.append(f"{prefix}:freeze_mtime_binding")
            freeze_at = datetime.fromtimestamp(
                freeze_mtime_ns / 1_000_000_000.0, timezone.utc
            )
            if not dispatch_starts or freeze_at >= min(dispatch_starts):
                blockers.append(f"{prefix}:freeze_not_before_dispatch")
            identity_checks = {
                "posterior_uid_pre": (
                    manifest.get("posterior_uid"),
                    selection.get("posterior_uid"),
                    lineage.get("posterior_uid_pre"),
                    prospective.get("posterior_uid_pre"),
                    receipt.get("posterior_uid_pre"),
                ),
                "posterior_uid_post": (
                    lineage.get("posterior_uid_post"),
                    prospective.get("posterior_uid_post"),
                    receipt.get("posterior_uid_post"),
                ),
                "world_action_uid": (
                    selection.get("world_action_uid"),
                    lineage.get("world_action_uid"),
                    prospective.get("world_action_uid"),
                    receipt.get("world_action_uid"),
                ),
                "information_action_uid": (
                    selection.get("information_action_uid"),
                    lineage.get("information_action_uid"),
                    prospective.get("information_action_uid"),
                    receipt.get("information_action_uid"),
                ),
                "result_uid": (
                    lineage.get("result_uid"),
                    prospective.get("result_uid"),
                    receipt.get("result_uid"),
                ),
            }
            for field, values in identity_checks.items():
                if len({str(value) for value in values if value not in (None, "")}) != 1:
                    blockers.append(f"{prefix}:{field}_mismatch")
            aggregate_bindings = {
                "input_manifest_ref": paths["input_manifest"],
                "selected_action_ref": paths["selection"],
                "round_receipt_ref": paths["receipt"],
            }
            for field, path in aggregate_bindings.items():
                if _repo_path(prospective.get(field, "")) != path.resolve():
                    blockers.append(f"{prefix}:prospective_{field}")
                hash_field = field.replace("_ref", "_sha256")
                if prospective.get(hash_field) != sha256_file(path):
                    blockers.append(f"{prefix}:prospective_{hash_field}")
            if prospective.get("round_receipt_uid") != receipt.get("receipt_uid"):
                blockers.append(f"{prefix}:prospective_receipt_uid")
            if not _truthy(prospective.get("selection_before_outcome")):
                blockers.append(f"{prefix}:prospective_selection_order")
            if _truthy(prospective.get("outcome_visible_before_freeze")):
                blockers.append(f"{prefix}:prospective_outcome_visibility")
            if not _truthy(prospective.get("freeze_before_dispatch_verified")):
                blockers.append(f"{prefix}:prospective_freeze_order")
            summaries.append(
                {
                    "policy_uid": policy_uid,
                    "round_uid": round_uid,
                    "receipt_uid": receipt.get("receipt_uid"),
                    "controller_timestamp_source": receipt.get(
                        "controller_timestamp_source"
                    ),
                }
            )
    return blockers, summaries


def validate(
    contract_path: Path = DEFAULT_CONTRACT,
    *,
    require_full_suite: bool = False,
) -> dict[str, Any]:
    contract = _load_yaml(contract_path)
    outputs = {key: _repo_path(value) for key, value in contract["outputs"].items()}
    required_keys = (
        "canonical_twin",
        "negative_result_freeze",
        "coverage_taxonomy",
        "support_domain",
        "tri_calibration",
        "policy_freeze",
        "control_contract",
        "prospective_manifest",
        "posterior_lineage",
        "tool_catalog",
        "decision_loop",
        "information_effects",
        "belief_coherence",
        "mechanism_gate",
        "reliability_gate",
        "failure_catalog",
        "architecture_gate",
        "recovery_catalog",
        "handoff",
    )
    blockers = _require_files({key: outputs[key] for key in required_keys})
    if require_full_suite:
        blockers.extend(_require_files({"full_suite": outputs["full_suite"]}))
    if blockers:
        return {
            "status": "failed",
            "blockers": blockers,
            "blocker_count": len(blockers),
        }

    canonical = json.loads(outputs["canonical_twin"].read_text(encoding="utf-8"))
    negative = json.loads(outputs["negative_result_freeze"].read_text(encoding="utf-8"))
    support = json.loads(outputs["support_domain"].read_text(encoding="utf-8"))
    calibration = json.loads(outputs["tri_calibration"].read_text(encoding="utf-8"))
    policy_freeze = json.loads(outputs["policy_freeze"].read_text(encoding="utf-8"))
    controls = json.loads(outputs["control_contract"].read_text(encoding="utf-8"))
    mechanism = json.loads(outputs["mechanism_gate"].read_text(encoding="utf-8"))
    reliability = json.loads(outputs["reliability_gate"].read_text(encoding="utf-8"))
    architecture = json.loads(outputs["architecture_gate"].read_text(encoding="utf-8"))
    recovery = json.loads(outputs["recovery_catalog"].read_text(encoding="utf-8"))
    handoff = json.loads(outputs["handoff"].read_text(encoding="utf-8"))

    if canonical.get("canonical_twin") != "U0_frozen":
        blockers.append("canonical_twin")
    if canonical.get("twin_parameter_update_allowed") is not False:
        blockers.append("twin_parameter_update_allowed")
    if canonical.get("observation_assimilation_allowed") is not True:
        blockers.append("observation_assimilation_allowed")
    if canonical.get("agent_parameter_update_allowed") is not False:
        blockers.append("agent_parameter_update_allowed")
    if negative.get("round24b_terminal_state") != "failed_prospective_twin_adaptation":
        blockers.append("negative_result_terminal_state")
    if negative.get("recommended_reference") != "U0_frozen":
        blockers.append("negative_result_reference")
    if int(negative.get("paired_scientific_rows", -1)) != int(
        contract["reference"]["expected_paired_row_count"]
    ):
        blockers.append("negative_result_paired_rows")
    for key in ("source_gate", "source_handoff", "source_paired_metrics"):
        binding = negative.get(key, {})
        path = _repo_path(binding.get("path", ""))
        if not path.is_file() or sha256_file(path) != binding.get("sha256"):
            blockers.append(f"negative_result_binding:{key}")

    taxonomy_rows = _read_tsv(outputs["coverage_taxonomy"])
    blockers.extend(validate_evidence_boundary(taxonomy_rows))
    if not taxonomy_rows or any(
        _truthy(row.get("unknown_rejection_reason")) for row in taxonomy_rows
    ):
        blockers.append("coverage_taxonomy_unclassified")
    if int(support.get("unknown_rejection_reason", -1)) != 0:
        blockers.append("support_unknown_rejection_reason")
    if support.get("coverage_floor_changed_from_test") is not False:
        blockers.append("coverage_floor_changed")
    if support.get("u0_operating_region_known") is not True:
        blockers.append("u0_operating_region")
    if float(support.get("coverage_floor", -1)) != float(
        contract["support_calibration"]["coverage_floor"]
    ):
        blockers.append("coverage_floor_identity")

    c1_cal = set(calibration.get("c1_calibration_units", []))
    c1_eval = set(calibration.get("c1_evaluation_units", []))
    c2s_cal = set(calibration.get("c2s_calibration_units", []))
    c2s_eval = set(calibration.get("c2s_evaluation_units", []))
    if not c1_cal or not c1_eval or c1_cal & c1_eval:
        blockers.append("c1_calibration_partition")
    if not c2s_cal or not c2s_eval or c2s_cal & c2s_eval:
        blockers.append("c2s_calibration_partition")
    if calibration.get("coverage_floor_changed_from_test") is not False:
        blockers.append("calibration_changed_floor")
    for domain in ("c1", "c2s"):
        if float(calibration[domain]["evaluation_coverage_adjusted"]) < float(
            contract["support_calibration"]["coverage_floor"]
        ):
            blockers.append(f"{domain}_calibrated_coverage")

    if set(policy_freeze.get("policies", {})) != set(POLICY_UIDS):
        blockers.append("policy_freeze_controls")
    if policy_freeze.get("policy_parameters_frozen_during_evaluation") is not True:
        blockers.append("policy_parameters_not_frozen")
    if controls.get("external_publication_baseline_substitute") is not False:
        blockers.append("internal_control_overclaim")

    prospective_rows = _read_tsv(outputs["prospective_manifest"])
    lineage_rows = _read_tsv(outputs["posterior_lineage"])
    tool_rows = _read_tsv(outputs["tool_catalog"])
    decision_rows = _read_tsv(outputs["decision_loop"])
    effect_rows = _read_tsv(outputs["information_effects"])
    coherence_rows = _read_tsv(outputs["belief_coherence"])
    failure_rows = _read_tsv(outputs["failure_catalog"])
    for rows, name in (
        (prospective_rows, "prospective"),
        (lineage_rows, "lineage"),
        (tool_rows, "tool"),
        (decision_rows, "decision"),
        (effect_rows, "effect"),
        (coherence_rows, "coherence"),
        (failure_rows, "failure"),
    ):
        boundary_blockers = validate_evidence_boundary(rows)
        blockers.extend(f"{name}:{value}" for value in boundary_blockers)
    expected_rows = len(POLICY_UIDS) * int(
        contract["static_policy"]["sequential_rounds"]
    )
    if len(prospective_rows) != expected_rows or len(lineage_rows) != expected_rows:
        blockers.append("prospective_round_count")
    blockers.extend(
        validate_posterior_chain(
            lineage_rows,
            expected_initial_posterior_uid=str(canonical["posterior_uid_pre"]),
            expected_rounds=int(contract["static_policy"]["sequential_rounds"]),
        )
    )
    round_blockers, round_summaries = validate_round_artifacts(
        run_root=_repo_path(contract["outputs"]["run_root"]),
        prospective_rows=prospective_rows,
        lineage_rows=lineage_rows,
        expected_rounds=int(contract["static_policy"]["sequential_rounds"]),
    )
    blockers.extend(round_blockers)
    if len(decision_rows) != expected_rows or len(effect_rows) != expected_rows:
        blockers.append("decision_effect_row_count")
    if len(coherence_rows) != expected_rows:
        blockers.append("coherence_row_count")
    for index, row in enumerate(effect_rows):
        try:
            tri_change = float(row.get("tri_change", "nan"))
            reliability_delta = float(row.get("reliability_proxy_delta", "nan"))
            reliability_pre = float(row.get("reliability_proxy_pre", "nan"))
            reliability_post = float(row.get("reliability_proxy_post", "nan"))
        except (TypeError, ValueError):
            blockers.append(f"effect_row_{index}:reliability_parse")
            continue
        if not all(
            value == value and abs(value) != float("inf")
            for value in (
                tri_change,
                reliability_delta,
                reliability_pre,
                reliability_post,
            )
        ):
            blockers.append(f"effect_row_{index}:reliability_nonfinite")
        if abs(reliability_delta - (reliability_post - reliability_pre)) > 1.0e-12:
            blockers.append(f"effect_row_{index}:reliability_delta")
        if abs(tri_change - reliability_delta) > 1.0e-12:
            blockers.append(f"effect_row_{index}:tri_change")
        if row.get("tri_change_semantics") != (
            "calibrated_reliability_proxy_delta_not_model_tri"
        ):
            blockers.append(f"effect_row_{index}:tri_change_semantics")
        if row.get("reliability_proxy_name") != RELIABILITY_PROXY_NAME:
            blockers.append(f"effect_row_{index}:reliability_proxy_name")
        if _truthy(row.get("tri_score_available")):
            blockers.append(f"effect_row_{index}:tri_score_availability")
    if any(not row.get("raw_bytes_hash") or not row.get("normalized_bytes_hash") for row in tool_rows if row.get("status") == "completed"):
        blockers.append("tool_result_hash")
    if sum(
        row.get("policy_uid") == "S_shared_belief_decision_risk"
        and _truthy(row.get("observation_admitted"))
        for row in lineage_rows
    ) < 3:
        blockers.append("shared_policy_admitted_rounds")
    required_failures = {
        "no_useful_query_available",
        "all_tools_unsupported",
        "tool_failure",
        "partial_result",
        "contradictory_evidence",
        "observation_increases_uncertainty",
        "observation_changes_ranking_incorrectly",
    }
    if not required_failures.issubset({row.get("scenario") for row in failure_rows}):
        blockers.append("failure_scenario_coverage")

    if architecture.get("status") not in TERMINAL_STATES:
        blockers.append("architecture_terminal_state")
    if mechanism.get("status") != architecture.get("status"):
        blockers.append("mechanism_architecture_status")
    if handoff.get("status") != architecture.get("status"):
        blockers.append("handoff_architecture_status")
    if reliability.get("insufficient_evidence_route_enabled") is not True:
        blockers.append("insufficient_evidence_route")
    if recovery.get("status") != "recovered_without_tool_redispatch":
        blockers.append("recovery_status")
    if recovery.get("registered_tool_redispatch_executed") is not False:
        blockers.append("recovery_tool_redispatch")
    if int(recovery.get("round_receipt_count", -1)) != expected_rows:
        blockers.append("recovery_receipt_count")
    if recovery.get("scientific_terminal_state") != architecture.get("status"):
        blockers.append("recovery_scientific_terminal_state")
    if recovery.get("tri_measurement", {}).get("model_tri_score_available") is not False:
        blockers.append("recovery_tri_availability")
    if recovery.get("tri_measurement", {}).get("reliability_proxy_name") != (
        RELIABILITY_PROXY_NAME
    ):
        blockers.append("recovery_reliability_proxy")
    blockers.extend(
        validate_recovery_catalog(
            recovery,
            outputs=outputs,
            expected_rounds=int(contract["static_policy"]["sequential_rounds"]),
        )
    )
    for payload, name in (
        (canonical, "canonical"),
        (negative, "negative"),
        (support, "support"),
        (calibration, "calibration"),
        (policy_freeze, "policy"),
        (controls, "controls"),
        (mechanism, "mechanism"),
        (reliability, "reliability"),
        (architecture, "architecture"),
        (recovery, "recovery"),
        (handoff, "handoff"),
    ):
        if payload.get("evidence_role") != "development_only":
            blockers.append(f"{name}:evidence_role")
        if payload.get("publication_claim_eligible") is not False:
            blockers.append(f"{name}:publication_claim_eligible")
        if payload.get("protected_holdout_access") != 0:
            blockers.append(f"{name}:protected_holdout_access")

    full_suite_summary: dict[str, Any] | None = None
    if require_full_suite:
        full_suite_summary = json.loads(outputs["full_suite"].read_text(encoding="utf-8"))
        if full_suite_summary.get("schema_version") != (
            "neuronsifter-static-loop-full-suite-execution-v1"
        ):
            blockers.append("full_suite_schema")
        if (
            full_suite_summary.get("status") != "passed"
            or full_suite_summary.get("full_suite_green") is not True
        ):
            blockers.append("full_suite_status")
        for field in ("tests_failed", "tests_errored", "tests_skipped"):
            if int(full_suite_summary.get(field, -1)) != 0:
                blockers.append(f"full_suite_{field}")
        collected = int(full_suite_summary.get("tests_collected", -1))
        passed = int(full_suite_summary.get("tests_passed", -1))
        if collected <= 0 or passed != collected:
            blockers.append("full_suite_test_count")
        accounting = full_suite_summary.get("slurm", {}).get(
            "terminal_accounting", {}
        )
        accounting_path = _repo_path(accounting.get("path", ""))
        if (
            full_suite_summary.get("slurm", {}).get("terminal_accounting_status")
            != "terminal_completed"
            or not accounting_path.is_file()
            or sha256_file(accounting_path) != accounting.get("sha256")
        ):
            blockers.append("full_suite_terminal_accounting")
        if architecture.get("full_suite_pending") is not False:
            blockers.append("architecture_full_suite_pending")
        if handoff.get("full_suite_pending") is not False:
            blockers.append("handoff_full_suite_pending")
        full_suite_binding = {
            "path": str(outputs["full_suite"].resolve().relative_to(REPO_ROOT.resolve())),
            "sha256": sha256_file(outputs["full_suite"]),
            "bytes": outputs["full_suite"].stat().st_size,
        }
        if architecture.get("validation", {}).get("full_suite_execution") != (
            full_suite_binding
        ):
            blockers.append("architecture_full_suite_binding")
        if handoff.get("full_suite_execution") != full_suite_binding:
            blockers.append("handoff_full_suite_binding")

    return {
        "schema_version": "neuronsifter-static-decision-loop-validation-v1",
        "status": "passed" if not blockers else "failed",
        "blockers": blockers,
        "blocker_count": len(blockers),
        "architecture_status": architecture.get("status"),
        "prospective_round_rows": len(prospective_rows),
        "tool_result_rows": len(tool_rows),
        "decision_rows": len(decision_rows),
        "round_receipts_validated": len(round_summaries),
        "round_receipts": round_summaries,
        "failure_rows": len(failure_rows),
        "full_suite": full_suite_summary,
        "protected_holdout_access": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--require-full-suite", action="store_true")
    args = parser.parse_args()
    result = validate(
        Path(args.contract).resolve(),
        require_full_suite=args.require_full_suite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
