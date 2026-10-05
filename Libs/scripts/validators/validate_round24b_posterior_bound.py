"""Fail-closed validation for the Round-24B posterior-bound evidence view.

The validator intentionally validates row-level lineage rather than trusting
aggregate counters written by the campaign runner.  It can be run before the
campaign (``--require-campaign`` omitted) to validate the contract/reference,
or after the campaign to validate all six blocker resolutions.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.models.modules.round24b_posterior_adapter import PosteriorBoundState
from Libs.models.modules.round24b_typed_actions import CurrentClampAction
from Libs.scripts.builders.build_round23_vnext_artifacts import (
    load_full_suite_execution,
)

CONTRACT = REPO_ROOT / "Libs/configs/neuronsifter/round24b_posterior_bound_evaluation.yaml"
REFERENCE = REPO_ROOT / "Output/audits/round24b_pre_update_reference.json"
POSTERIOR = REPO_ROOT / "Output/audits/round24b_pre_update_posterior.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _is_false(value: Any) -> bool:
    return value is False or str(value).strip().lower() in {"false", "0"}


def _require_boundary(payload: Mapping[str, Any], name: str) -> None:
    if payload.get("evidence_role") != "development_only":
        raise ValueError(f"{name} crossed the development evidence boundary")
    if not _is_false(payload.get("publication_claim_eligible")):
        raise ValueError(f"{name} is publication-claim eligible")
    if int(payload.get("protected_holdout_access", payload.get("protected_test_access", -1))) != 0:
        raise ValueError(f"{name} accessed a protected holdout")


def _validate_contract() -> dict[str, Any]:
    payload = yaml.safe_load(CONTRACT.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "neuronsifter-round24b-posterior-bound-evaluation-v1":
        raise ValueError("Round-24B contract schema version drift")
    _require_boundary(payload["evidence_boundary"], "Round-24B contract boundary")
    if payload["updates"].get("agent_parameters_update_allowed") is not False:
        raise ValueError("Round-24B Agent parameters are not frozen")
    if payload["updates"].get("neural_checkpoint_update_allowed") is not False:
        raise ValueError("Round-24B neural checkpoints are not frozen")
    if int(payload["prospective"].get("rounds", 0)) < 3:
        raise ValueError("Round-24B requires at least three prospective rounds")
    if int(payload["statistics"].get("bootstrap_replicates", 0)) != 2000:
        raise ValueError("Round-24B bootstrap contract drift")
    if int(payload["statistics"].get("bootstrap_seed", -1)) != 3407:
        raise ValueError("Round-24B bootstrap seed drift")
    if float(payload["updates"].get("minimum_new_evidence_fit_gain", -1.0)) != 0.0:
        raise ValueError("Round-24B fresh-evidence fit threshold drift")
    typed = payload["typed_current_clamp"]
    response = list(typed.get("response_species_ids", []))
    signed = list(typed.get("signed_species_ids", []))
    if not response or len(set(response)) != len(response) or not set(signed).issubset(set(response)):
        raise ValueError("Round-24B typed current-clamp species contract is invalid")
    channels = typed.get("posterior_condition_channel_species")
    if not isinstance(channels, Mapping) or set(channels) != {"posterior_response_mean", "posterior_response_std"}:
        raise ValueError("Round-24B posterior response channel mapping is incomplete")
    channel_values = [str(channels[key]) for key in ("posterior_response_mean", "posterior_response_std")]
    if len(set(channel_values)) != 2 or not set(channel_values).issubset(set(response)):
        raise ValueError("Round-24B posterior response channels must be distinct declared response species")
    return payload


def _validate_reference(contract: Mapping[str, Any]) -> dict[str, Any]:
    if not REFERENCE.is_file() or not POSTERIOR.is_file():
        raise ValueError("Round-24B pre-update reference is not materialized")
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    posterior = json.loads(POSTERIOR.read_text(encoding="utf-8"))
    _require_boundary(reference, "Round-24B reference")
    if reference.get("status") != "frozen" or reference.get("pre_update_reference_frozen") is not True:
        raise ValueError("Round-24B reference is not frozen")
    if reference.get("posterior_uid_pre") != posterior.get("posterior_uid"):
        raise ValueError("Round-24B posterior UID does not match reference")
    if reference.get("posterior_record", {}).get("sha256") != _sha256(POSTERIOR):
        raise ValueError("Round-24B posterior record hash drift")
    if reference.get("checkpoint_hashes_verified") is not True:
        raise ValueError("Round-24B checkpoint hashes were not verified")
    if reference.get("agent_parameters_frozen") is not True:
        raise ValueError("Round-24B Agent freeze is absent")
    return reference


def _validate_binding_rows(reference: Mapping[str, Any]) -> dict[str, Any]:
    c1_path = REPO_ROOT / "Output/audits/round24b_c1_posterior_binding.json"
    c2_path = REPO_ROOT / "Output/audits/round24b_c2s_posterior_action_binding.json"
    if not c1_path.is_file() or not c2_path.is_file():
        raise FileNotFoundError("Round-24B binding audits are missing")
    c1 = json.loads(c1_path.read_text(encoding="utf-8"))
    c2 = json.loads(c2_path.read_text(encoding="utf-8"))
    _require_boundary(c1, "C1 binding audit")
    _require_boundary(c2, "C2-S binding audit")
    if c1.get("unbound_c1_metric_rows") != 0 or c1.get("stale_posterior_rows") != 0:
        raise ValueError("C1 binding audit contains unbound or stale rows")
    if c1.get("unacknowledged_model_consumption_rows") != 0 or c1.get("zero_effect_probe_rows") != 0:
        raise ValueError("C1 binding audit does not prove direct posterior model consumption")
    if c2.get("implicit_action_rows") != 0 or c2.get("stale_posterior_rows") != 0:
        raise ValueError("C2-S binding audit contains implicit or stale rows")
    if c2.get("unacknowledged_model_consumption_rows") != 0 or c2.get("zero_effect_probe_rows") != 0:
        raise ValueError("C2-S binding audit does not prove direct posterior model consumption")
    if c2.get("prospective_metric_row_count", 0) <= 0:
        raise ValueError("C2-S prospective posterior-action metric rows are absent")
    if c2.get("prospective_unbound_rows") != 0:
        raise ValueError("C2-S prospective posterior-action rows are not bound")
    if c2.get("prospective_action_bound_rows") != c2.get("prospective_metric_row_count"):
        raise ValueError("C2-S prospective posterior-action rows lack typed world actions")
    if c2.get("split_overlap") != 0:
        raise ValueError("C2-S binding audit reports split overlap")
    return {"c1": c1, "c2s": c2}


def _validate_action_and_dispatch_manifests(contract: Mapping[str, Any]) -> dict[str, Any]:
    prospective_path = REPO_ROOT / "Input/manifests/round24b/prospective_information_actions.tsv"
    action_path = REPO_ROOT / "Input/manifests/round24b/c2s_world_actions.tsv"
    rows = _load_tsv(prospective_path)
    action_rows = _load_tsv(action_path)
    if not rows or not action_rows or len(rows) != len(action_rows):
        raise ValueError("Round-24B prospective and typed-action manifest counts differ")
    action_fields = tuple(action_rows[0])
    projected_rows = [
        {field: row.get(field, "") for field in action_fields} for row in rows
    ]
    if projected_rows != action_rows:
        raise ValueError("Round-24B prospective and typed-action identities differ")
    required = {
        "world_action_uid", "action_type", "target", "stimulation_site", "amplitude_nA",
        "schedule", "delay_ms", "duration_ms", "context", "producer_uid", "producer_version",
    }
    historical_manifest = REPO_ROOT / contract["reference"]["c2s_pool_path"]
    historical = json.loads(historical_manifest.read_text(encoding="utf-8"))
    historical_protocols = {
        (
            float(row["action"]["dose_value"]),
            float(row["action"]["duration_ms"]),
            float(row["action"]["delay_ms"]),
        )
        for row in historical["records"]
        if row["action"]["action_role"] == "intervention"
    }
    for row in rows:
        _require_boundary(row, f"prospective action {row.get('action_uid')}")
        missing = sorted(field for field in required if not row.get(field))
        if missing:
            raise ValueError(f"Round-24B typed action row lacks {missing}")
        schedule = json.loads(row["schedule"])
        role = str(row["action_role"])
        action = CurrentClampAction(
            action_uid=str(row["action_uid"]),
            pair_uid=str(row["pair_uid"]),
            role=role,
            amplitude_nA=float(row["amplitude_nA"]),
            duration_ms=float(row["duration_ms"]),
            delay_ms=float(row["delay_ms"]),
            source_model_id=str(row["producer_uid"]),
            split_role=str(row["split_role"]),
        )
        if row["world_action_uid"] != action.to_world_action().action_uid or schedule != action.schedule_payload():
            raise ValueError(f"Round-24B typed action identity drift: {row.get('action_uid')}")
        for field in (
            "prospective_unit_uid",
            "posterior_uid",
            "parent_posterior_uid",
            "round_unit_manifest_ref",
            "round_unit_manifest_sha256",
        ):
            if not row.get(field):
                raise ValueError(
                    f"Round-24B prospective lineage lacks {field}: {row.get('action_uid')}"
                )
        if row["posterior_uid"] != row["parent_posterior_uid"]:
            raise ValueError(
                f"Round-24B prospective parent identity drift: {row.get('action_uid')}"
            )
        if row.get("requested_before_outcome") not in {"True", "true", "1"}:
            raise ValueError(
                f"Round-24B prospective action was not frozen before outcome: {row.get('action_uid')}"
            )
        round_manifest = REPO_ROOT / row["round_unit_manifest_ref"]
        if not round_manifest.is_file() or _sha256(round_manifest) != row["round_unit_manifest_sha256"]:
            raise ValueError(
                f"Round-24B prospective round manifest hash drift: {row.get('action_uid')}"
            )
        protocol = (action.amplitude_nA, action.duration_ms, action.delay_ms)
        if role == "intervention" and (protocol in historical_protocols or protocol == (0.2, 120.0, 40.0)):
            raise ValueError(f"Round-24B prospective protocol overlaps historical/protected identity: {protocol}")
    expected = int(contract["prospective"]["rounds"]) * 2
    if len(rows) != expected or len({row["action_uid"] for row in rows}) != expected:
        raise ValueError("Round-24B typed action manifest has incomplete or duplicate catalog units")
    dispatch_path = REPO_ROOT / contract["outputs"]["output_root"] / "prospective_dispatch_manifest.json"
    if dispatch_path.is_file():
        dispatch = json.loads(dispatch_path.read_text(encoding="utf-8"))
        _require_boundary(dispatch, "Round-24B dispatch manifest")
        records = dispatch.get("records", [])
        if len(records) != int(contract["prospective"]["rounds"]):
            raise ValueError("Round-24B dispatch record count does not match sequential rounds")
        common_parents = set()
        for record in records:
            common_parents.add(str(record.get("parent_posterior_uid", "")))
            timestamps = [record.get("dispatch_started_at"), record.get("dispatch_finished_at")]
            if any(not value for value in timestamps):
                raise ValueError("Round-24B dispatch record lacks execution timestamps")
            if datetime.fromisoformat(str(timestamps[1])) < datetime.fromisoformat(str(timestamps[0])):
                raise ValueError("Round-24B dispatch finished before it started")
        if "" in common_parents:
            raise ValueError("Round-24B dispatch record lacks parent posterior identity")
        if not all(record.get("common_stream") is True for record in records):
            raise ValueError("Round-24B dispatch records are not marked as common evidence")
    return {"prospective_action_rows": len(rows), "fresh_protocol_overlap": 0, "dispatch_records": len(dispatch_path.is_file() and records or [])}


def _validate_posterior_catalog(contract: Mapping[str, Any]) -> dict[str, Any]:
    path = REPO_ROOT / contract["outputs"]["posterior_lineage_catalog"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require_boundary(payload, "Round-24B posterior lineage catalog")
    records = payload.get("posteriors", [])
    states = {row["posterior_uid"]: PosteriorBoundState.from_record(row) for row in records}
    if len(states) != len(records) or payload.get("posterior_count") != len(states):
        raise ValueError("Round-24B posterior lineage catalog has duplicate identities")
    for state in states.values():
        if state.parent_posterior_uid and state.parent_posterior_uid not in states:
            raise ValueError(f"Round-24B posterior parent is absent: {state.posterior_uid}")
    if int(payload.get("stale_metric_posterior_rows", -1)) != 0:
        raise ValueError("Round-24B posterior lineage catalog reports stale metric rows")
    return {"posterior_count": len(states), "lineage_closed": True}


def _validate_metric_rows(reference: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    path = REPO_ROOT / "Output/runs/round24b/scientific_metric_inputs.tsv"
    rows = _load_tsv(path)
    if not rows:
        raise ValueError("Round-24B scientific metric input is empty")
    pre_uids = {str(reference["posterior_uid_pre"])}
    posterior_uids = {str(row.get("posterior_uid", "")) for row in rows}
    if not posterior_uids.issuperset(pre_uids) or "" in posterior_uids:
        raise ValueError("Round-24B metric rows have missing posterior identities")
    for row in rows:
        _require_boundary(row, f"metric row {row.get('evaluation_uid')}")
        if row.get("finite") not in {"True", "true", "1"}:
            raise ValueError(f"Round-24B metric row is non-finite: {row.get('evaluation_uid')}")
        ref = row.get("metric_input_ref", "")
        if not ref or not (REPO_ROOT / ref).is_file():
            raise ValueError(f"Round-24B metric input reference is missing: {ref}")
        snapshot = json.loads((REPO_ROOT / ref).read_text(encoding="utf-8"))
        if snapshot.get("posterior_uid") != row.get("posterior_uid"):
            raise ValueError(f"Round-24B snapshot posterior mismatch: {ref}")
        checkpoint_hashes = {
            snapshot.get("posterior_checkpoint_hash"),
            snapshot.get("posterior_source_checkpoint_sha256"),
            snapshot.get("world_action_checkpoint_sha256"),
            snapshot.get("encoder_checkpoint_sha256"),
        }
        if row.get("posterior_checkpoint_hash") not in checkpoint_hashes:
            raise ValueError(f"Round-24B snapshot checkpoint mismatch: {ref}")
        domain = str(row.get("domain", ""))
        if domain in {"c1", "c2s"}:
            if row.get("posterior_direct_model_consumption") not in {True, "True", "true", "1"}:
                raise ValueError(f"Round-24B metric row does not acknowledge direct posterior model consumption: {row.get('evaluation_uid')}")
            try:
                effect = float(row.get("posterior_model_path_effect_abs", "nan"))
            except (TypeError, ValueError):
                effect = float("nan")
            if not effect > 0.0:
                raise ValueError(f"Round-24B metric row has no positive model-path effect probe: {row.get('evaluation_uid')}")
        if domain in {"c2s", "c2s_prospective"}:
            for field in ("world_action_uid", "control_world_action_uid", "initial_state_uid", "trajectory_uid", "producer_uid"):
                if not row.get(field):
                    raise ValueError(f"C2-S row lacks {field}: {row.get('evaluation_uid')}")
        if domain == "c2s_prospective":
            if row.get("posterior_action_response_direct_consumption") not in {True, "True", "true", "1"}:
                raise ValueError(f"C2-S prospective row lacks posterior action-response binding: {row.get('evaluation_uid')}")
            if row.get("fresh_observation_reused_for_fit_evaluation") not in {True, "True", "true", "1"}:
                raise ValueError(f"C2-S prospective row lacks fresh-observation status: {row.get('evaluation_uid')}")
            if row.get("generalization_claim_eligible") not in {False, "False", "false", "0"}:
                raise ValueError(f"C2-S prospective row is marked generalization-eligible: {row.get('evaluation_uid')}")
    paired_path = REPO_ROOT / contract["outputs"]["paired_metrics"]
    paired = _load_tsv(paired_path)
    if not paired:
        raise ValueError("Round-24B paired metric report is empty")
    keys = set()
    for row in paired:
        _require_boundary(row, f"paired row {row.get('evaluation_uid')}")
        key = (row.get("strategy_uid"), row.get("round_uid"), row.get("evaluation_uid"), row.get("metric_name"))
        if key in keys:
            raise ValueError(f"Duplicate paired metric unit: {key}")
        keys.add(key)
        if row.get("posterior_uid_pre") == "" or row.get("posterior_uid_post") == "":
            raise ValueError(f"Paired row lacks pre/post posterior identity: {key}")
        if row.get("finite") not in {"True", "true", "1"}:
            raise ValueError(f"Paired row is non-finite: {key}")
        for field in ("metric_input_ref_pre", "metric_input_ref_post"):
            ref = row.get(field, "")
            if not ref or not (REPO_ROOT / ref).is_file():
                raise ValueError(f"Paired row lacks metric snapshot: {key}/{field}")
        pre = json.loads((REPO_ROOT / row["metric_input_ref_pre"]).read_text(encoding="utf-8"))
        post = json.loads((REPO_ROOT / row["metric_input_ref_post"]).read_text(encoding="utf-8"))
        if pre.get("posterior_uid") != row["posterior_uid_pre"] or post.get("posterior_uid") != row["posterior_uid_post"]:
            raise ValueError(f"Paired row posterior/snapshot mismatch: {key}")
        for snapshot, label in ((pre, "pre"), (post, "post")):
            units = snapshot.get("evaluation_units")
            if units and row["evaluation_unit_uid"] not in units:
                raise ValueError(f"Paired evaluation unit absent from {label} snapshot: {key}")
    return {
        "metric_row_count": len(rows),
        "paired_row_count": len(paired),
        "posterior_uid_count": len(posterior_uids),
        "direct_model_metric_rows": sum(row.get("domain") in {"c1", "c2s"} for row in rows),
        "prospective_action_response_metric_rows": sum(row.get("domain") == "c2s_prospective" for row in rows),
    }


def _validate_update_controls(contract: Mapping[str, Any]) -> dict[str, Any]:
    freeze_path = REPO_ROOT / "Output/audits/round24b_update_strategy_freeze.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    _require_boundary(freeze, "Round-24B update strategy freeze")
    expected = {str(row["uid"]): str(row["mode"]) for row in contract["updates"]["strategies"]}
    if set(freeze.get("strategies", [])) != set(expected) or freeze.get("common_prospective_stream") is not True:
        raise ValueError("Round-24B update strategy freeze is incomplete")
    operation_count = 0
    previous_frozen_at: dict[str, datetime] = {}
    common_round_parents: dict[str, str] = {}
    for strategy_uid, mode in expected.items():
        for round_index in range(1, int(contract["prospective"]["rounds"]) + 1):
            round_uid = f"round_{round_index}"
            round_dir = REPO_ROOT / contract["outputs"]["output_root"] / strategy_uid / round_uid
            required_files = (
                f"{round_uid}_input_manifest.json",
                f"{round_uid}_selected_information_actions.tsv",
                f"{round_uid}_tool_results.tsv",
                f"{round_uid}_evidence_admission.tsv",
                f"{round_uid}_posterior_update.json",
                f"{round_uid}_twin_candidate_update.json",
                f"{round_uid}_retention.tsv",
                f"{round_uid}_calibration.tsv",
                f"{round_uid}_promotion_or_rollback.json",
                f"{round_uid}_report.json",
            )
            missing = [name for name in required_files if not (round_dir / name).is_file()]
            if missing:
                raise ValueError(f"Round-24B per-round artifacts are missing: {strategy_uid}/{round_uid}/{missing}")
            path = round_dir / f"{round_uid}_posterior_update.json"
            update = json.loads(path.read_text(encoding="utf-8"))
            _require_boundary(update, f"Round-24B update {strategy_uid}/round_{round_index}")
            if update.get("strategy_uid") != strategy_uid or update.get("update_mode") != mode:
                raise ValueError(f"Round-24B update strategy/mode drift: {path}")
            if update.get("agent_parameters_changed") is not False:
                raise ValueError(f"Round-24B Agent parameters changed: {path}")
            input_manifest = json.loads((round_dir / f"{round_uid}_input_manifest.json").read_text(encoding="utf-8"))
            selected_rows = _load_tsv(round_dir / f"{round_uid}_selected_information_actions.tsv")
            if len(selected_rows) != 1:
                raise ValueError(f"Round-24B selected action freeze is malformed: {round_dir}")
            frozen_at = input_manifest.get("frozen_at")
            if not frozen_at or selected_rows[0].get("frozen_at") != frozen_at:
                raise ValueError(f"Round-24B input/action freeze timestamp mismatch: {round_dir}")
            current_time = datetime.fromisoformat(str(frozen_at))
            if strategy_uid in previous_frozen_at and current_time <= previous_frozen_at[strategy_uid]:
                raise ValueError(f"Round-24B round freeze order is not strictly increasing: {round_dir}")
            previous_frozen_at[strategy_uid] = current_time
            if input_manifest.get("posterior_uid") != input_manifest.get("assimilation_parent_posterior_uid"):
                raise ValueError(f"Round-24B frozen input parent identity drift: {round_dir}")
            if update.get("parent_posterior_uid") != input_manifest.get("posterior_uid"):
                raise ValueError(f"Round-24B update parent does not match frozen input: {path}")
            selection_parent = str(input_manifest.get("selection_posterior_uid", ""))
            common_round_parents.setdefault(round_uid, selection_parent)
            if common_round_parents[round_uid] != selection_parent:
                raise ValueError(f"Round-24B common selection posterior drift: {round_dir}")
            for operation in update.get("decision_consistency_operations", {}).values():
                identities = {
                    str(operation[field])
                    for field in (
                        "posterior_uid_rollout",
                        "posterior_uid_query",
                        "posterior_uid_assimilation_parent",
                        "posterior_uid_decision",
                    )
                }
                if len(identities) != 1:
                    raise ValueError(f"Round-24B decision-consistency failure: {path}")
                operation_count += 1
    return {"strategy_count": len(expected), "round_count_per_strategy": int(contract["prospective"]["rounds"]), "validated_decision_operations": operation_count, "frozen_round_order_validated": True, "common_selection_parent_rounds": len(common_round_parents)}


def validate(require_campaign: bool = True) -> dict[str, Any]:
    contract = _validate_contract()
    reference = _validate_reference(contract)
    result: dict[str, Any] = {
        "schema_version": "neuronsifter-round24b-posterior-bound-validation-v1",
        "contract_valid": True,
        "pre_update_reference_frozen": True,
        "protected_holdout_access": 0,
        "publication_comparison_open": False,
        "evidence_role": "development_only",
        "publication_claim_eligible": False,
    }
    if not require_campaign:
        result["status"] = "reference_valid"
        return result
    binding = _validate_binding_rows(reference)
    actions = _validate_action_and_dispatch_manifests(contract)
    lineage = _validate_posterior_catalog(contract)
    metrics = _validate_metric_rows(reference, contract)
    controls = _validate_update_controls(contract)
    round23_full_suite = load_full_suite_execution()
    if round23_full_suite.get("successor_source_binding_valid") is not True:
        raise ValueError("Round-24B source changes are not bound to the Round-23 lineage")
    gate_path = REPO_ROOT / contract["outputs"]["twin_gate"]
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    _require_boundary(gate, "Round-24B Twin Gate")
    if gate.get("prospective_rounds", 0) < 3:
        raise ValueError("Round-24B Twin Gate has fewer than three rounds")
    if gate.get("decision_consistency", {}).get("preserved") is not True or gate.get("decision_consistency", {}).get("hidden_scientific_decision_states") != 0:
        raise ValueError("Round-24B decision-consistency Gate is not closed")
    if gate.get("agent", {}).get("parameters_updated") is not False or gate.get("agent", {}).get("adaptation_executed") is not False:
        raise ValueError("Round-24B Agent adaptation was executed or changed parameters")
    expected_blockers = {
        "c1_posterior_not_bound_to_encoder_evaluation",
        "c2s_posterior_not_bound_to_world_action_rollout",
        "world_action_uid_absent_from_prospective_units",
        "pre_post_scientific_metric_inputs_absent",
        "update_strategy_controls_not_executed",
        "conditional_agent_execution_invalidated_by_premature_gate",
    }
    blockers = gate.get("six_blockers", {})
    if set(blockers) != expected_blockers or any(value != "resolved" for value in blockers.values()):
        raise ValueError("Round-24B six-blocker terminal state is incomplete")
    if gate.get("publication", {}).get("comparison_open") is not False or int(gate.get("publication", {}).get("protected_holdout_access", -1)) != 0:
        raise ValueError("Round-24B publication boundary is open")
    rollback = json.loads((REPO_ROOT / "Output/audits/round24b_rollback_restoration.json").read_text(encoding="utf-8"))
    _require_boundary(rollback, "Round-24B rollback restoration")
    if not all(rollback.get(key) is True for key in ("rollback_executed", "exact_prior_checkpoint_restored", "posterior_identity_restored", "metrics_restored_within_tolerance")):
        raise ValueError("Round-24B rollback restoration is incomplete")
    result.update({"status": "passed", "binding": binding, "actions": actions, "lineage": lineage, "metrics": metrics, "controls": controls, "round23_successor_source_binding_valid": True, "twin_gate_status": gate.get("status")})
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate(require_campaign=not args.reference_only), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
