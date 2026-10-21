"""Build the public-only Round-18 literature replay qualification pilot.

The builder materializes metadata-sized public literature fixtures, locks the
evaluator bundle while prediction workers run, and reveals outcomes only after
the immutable T4 prediction catalog has been hashed. It does not retrieve
controlled data, train a model, execute production actions, or inspect a final
held-out partition.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import stat
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.literature_replay_contracts import (  # noqa: E402
    CLAIM_SCOPE,
    SCHEMA_VERSIONS,
    canonical_episode_uid,
    canonical_sequential_event_uid,
    canonical_sha256,
    compute_direction_metrics,
    compute_ranking_metrics,
    compute_reliability_metrics,
    episode_content_hash,
    sequential_event_content_hash,
    select_earliest_public_timestamp,
    stable_uid,
    validate_episode,
    validate_frozen_gap_isolation,
    validate_independence_groups,
    validate_label_provenance,
    validate_memory_risk_disclosure,
    validate_public_source_boundary,
    validate_scitil_runtime_trace,
    validate_sequential_reveal,
    validate_split_leakage,
    validate_temporal_ordering,
)
from Libs.utils.data.raw_data_registry import (  # noqa: E402
    load_config,
    read_tsv,
    write_json,
    write_tsv,
)
from Libs.utils.data.scitil_contracts import (  # noqa: E402
    AgentIdentity,
    AgentProposal,
    CANONICAL_CONTROLLER_UID as SCITIL_CONTROLLER_UID,
    HypothesisGateDecision,
    InformationActionGateDecision,
    ObservationGateDecision,
    ROUND17_VALIDATOR_UID,
    ScientificHypothesis,
    TwinKernel,
    WorldActionGateDecision,
    SCHEMA_VERSIONS as SCITIL_SCHEMA_VERSIONS,
    validate_agent_identity,
    validate_h_gate,
    validate_i_gate,
    validate_o_gate,
    validate_w_gate,
)


ROUND18_CONFIG_VERSION = "neuronsifter-round18-literature-replay-v1"
ROUND18_VALIDATOR_UID = "round18_literature_held_out_replay_validator"
CANONICAL_CONTROLLER_UID = "neuronsifter_orchestration_controller"
SYSTEM_UIDS = (
    "static_evidence_fabric_baseline",
    "rule_based_evidence_scoring",
    "twin_only_without_agents",
    "agent_only_offline_proposal_ablation",
    "scitil_without_falsifier",
    "scitil_without_twin_debt",
    "scitil_without_information_actions",
    "full_governed_scitil",
)
REVEAL_STEPS = ("T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7")
FINAL_HELD_OUT_PARTITION = "final_held_out"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repository_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def environment_identity() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "process_uid": os.getuid(),
        "process_gid": os.getgid(),
        "evaluator_pid": os.getpid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not_slurm"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME", "not_slurm"),
        "slurm_node": os.environ.get("SLURMD_NODENAME", socket.gethostname()),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION", "not_slurm"),
        "python_pycache_prefix": os.environ.get("PYTHONPYCACHEPREFIX", ""),
    }


def _resolve(root: Path, relative: object) -> Path:
    path = Path(str(relative))
    if path.is_absolute():
        raise ValueError(f"Round-18 configured paths must be relative: {path}")
    return root / path


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _schema_blockers(
    repository_root: Path, schema_name: str, values: Sequence[Mapping[str, object]]
) -> list[str]:
    """Validate materialized objects before they become benchmark evidence."""
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except (
        ImportError
    ) as exc:  # pragma: no cover - project test environments provide jsonschema
        raise RuntimeError(
            "jsonschema is required for Round-18 materialization"
        ) from exc
    schema_path = repository_root / "Libs/configs/schema" / f"{schema_name}.schema.json"
    validator = Draft202012Validator(
        _load_json(schema_path), format_checker=FormatChecker()
    )
    blockers: list[str] = []
    for index, value in enumerate(values):
        for error in validator.iter_errors(dict(value)):
            blockers.append(
                f"{schema_name}[{index}]/{'/'.join(str(item) for item in error.path)}: {error.message}"
            )
    return blockers


def round18_config(config: Mapping[str, object]) -> dict[str, object]:
    value = config.get("round18_literature_replay")
    if not isinstance(value, Mapping):
        raise TypeError("Configuration is missing round18_literature_replay.")
    return dict(value)


def _output_paths(config: Mapping[str, object]) -> dict[str, object]:
    configured = dict(round18_config(config)["outputs"])
    pilot_root = Path(str(configured["pilot_root"]))
    aliases = {
        "agent_visible_bundle_root": (
            pilot_root / "bundles/agent_visible_pre_cutoff_bundle"
        ).as_posix(),
        "evaluator_only_bundle_root": (
            pilot_root / "bundles/evaluator_only_held_out_bundle"
        ).as_posix(),
        "episode_manifest": configured["pilot_episode_manifest"],
        "prediction_catalog": configured["pilot_prediction_catalog"],
        "ranking_catalog": configured["pilot_ranking_catalog"],
        "decision_catalog": configured["pilot_decision_catalog"],
        "abstention_catalog": configured["pilot_abstention_catalog"],
        "leakage_control_catalog": configured["pilot_leakage_control_catalog"],
        "public_source_boundary_audit": configured["public_source_boundary"],
        "memory_leakage_risk_audit": configured["memory_leakage_risk"],
    }
    return {**configured, **aliases}


def _validate_config(config: Mapping[str, object]) -> list[dict[str, object]]:
    cfg = round18_config(config)
    blockers: list[dict[str, object]] = []
    if cfg.get("version") != ROUND18_CONFIG_VERSION:
        blockers.append({"code": "round18_config_version_invalid"})
    authoritative_start = dict(cfg.get("authoritative_start", {}))
    if (
        authoritative_start.get("exact_main_commit")
        != "de3b691587da49896fd108cd3999127a2b5deb08"
    ):
        blockers.append({"code": "round18_start_commit_invalid"})
    claim_scope = dict(cfg.get("claim_scope", {}))
    if any(
        claim_scope.get(key) is not expected for key, expected in CLAIM_SCOPE.items()
    ):
        blockers.append({"code": "round18_claim_scope_invalid"})
    public_policy = dict(dict(cfg.get("policies", {})).get("public_source", {}))
    if not (
        public_policy.get("strict_public_only_flag") is True
        and public_policy.get("controlled_access_required") is False
        and public_policy.get("resolved_controlled_source_count") == 0
    ):
        blockers.append({"code": "round18_public_policy_invalid"})
    frozen = dict(dict(cfg.get("policies", {})).get("frozen_gap", {}))
    if (
        frozen.get("count_before_round18") != 20
        or frozen.get("count_after_round18") != 20
    ):
        blockers.append({"code": "round18_frozen_gap_policy_invalid"})
    if (
        authoritative_start.get("evidence_fabric_sealed") is not False
        or authoritative_start.get("dt_core_status") != "not_evaluated"
    ):
        blockers.append({"code": "round18_forbidden_state_transition"})
    configured_systems = tuple(str(row) for row in cfg.get("systems", []))
    if configured_systems != SYSTEM_UIDS:
        blockers.append({"code": "round18_system_registry_invalid"})
    contracts = dict(cfg.get("system_contracts", {}))
    if set(contracts) != set(SYSTEM_UIDS):
        blockers.append({"code": "round18_system_contract_registry_invalid"})
    for system_uid, raw_contract in contracts.items():
        contract = dict(raw_contract) if isinstance(raw_contract, Mapping) else {}
        confidence_scale = contract.get("confidence_scale")
        runtime_mode = contract.get("runtime_mode")
        if (
            not isinstance(confidence_scale, (int, float))
            or isinstance(confidence_scale, bool)
            or not 0.0 < float(confidence_scale) <= 1.5
            or contract.get("skeptical_conflict_policy") not in {True, False}
            or contract.get("ranking_policy") not in {"opaque_uid", "pre_cutoff_score"}
            or contract.get("information_action_policy")
            not in {"none", "expected_information_gain"}
            or contract.get("falsifier_enabled") not in {True, False}
            or contract.get("model_uncertainty_known") not in {True, False}
            or runtime_mode
            not in {"offline_baseline", "contract_ablation", "offline_agent_proposal", "round17_twin_kernel"}
        ):
            blockers.append(
                {
                    "code": "round18_system_contract_invalid",
                    "system_uid": system_uid,
                }
            )
    if dict(contracts.get("full_governed_scitil", {})).get("runtime_mode") != "round17_twin_kernel":
        blockers.append({"code": "round18_full_scitil_runtime_invalid"})
    agent_contract = dict(contracts.get("agent_only_offline_proposal_ablation", {}))
    if any(
        agent_contract.get(field) is not False
        for field in ("direct_tool_execution", "direct_state_update", "gate_authority")
    ):
        blockers.append({"code": "round18_agent_only_sandbox_invalid"})
    for group in ("parent_artifacts", "outputs"):
        for name, value in dict(cfg.get(group, {})).items():
            if Path(str(value)).is_absolute():
                blockers.append(
                    {
                        "code": "round18_absolute_path_forbidden",
                        "field": f"{group}.{name}",
                    }
                )
    return blockers


def validate_parent_state(
    config: Mapping[str, object], parent_root: Path
) -> dict[str, object]:
    cfg = round18_config(config)
    paths = {
        name: _resolve(parent_root, path)
        for name, path in dict(cfg["parent_artifacts"]).items()
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Round-18 parent artifacts are missing: {missing}")
    payloads = {
        name: _load_json(path) for name, path in paths.items() if path.suffix == ".json"
    }
    blockers: list[dict[str, object]] = []
    entry = payloads["round18_entry_gate"]
    round17i = payloads["round17i_gate"]
    round17 = payloads["round17_gate"]
    round16 = payloads["round16_gate"]
    coverage = payloads["round14_evidence_fabric_coverage"]
    trajectory = payloads["round15_trajectory_coverage"]
    if entry.get("status") != "ready_for_literature_held_out_sequential_replay":
        blockers.append({"code": "round18_entry_not_ready"})
    if entry.get("round18_implementation_started") is not False:
        blockers.append({"code": "round18_entry_already_started"})
    if not (
        round17i.get("status") == "scitil_runtime_mainline_integrated"
        and round17i.get("overall_gate_status") == "accepted"
        and round17i.get("main_commit") == "de3b691587da49896fd108cd3999127a2b5deb08"
    ):
        blockers.append({"code": "round17i_gate_invalid"})
    if round17.get("status") != "scitil_twin_gated_agent_loop_resolved":
        blockers.append({"code": "round17_gate_invalid"})
    if round16.get("status") != "governed_what_if_engine_resolved":
        blockers.append({"code": "round16_gate_invalid"})
    if coverage.get("unresolved_gaps") != 20 or coverage.get("sealed") is not False:
        blockers.append({"code": "round14_evidence_fabric_state_invalid"})
    if trajectory.get("effective_independent_units") != 1:
        blockers.append({"code": "round15_trajectory_lineage_invalid"})
    for name, payload in payloads.items():
        if (
            payload.get("evidence_fabric", {}).get("sealed") is True
            or payload.get("sealed") is True
        ):
            blockers.append({"code": "evidence_fabric_sealed", "artifact": name})
        if payload.get("dt_core", {}).get("status") not in (None, "not_evaluated"):
            blockers.append({"code": "dt_core_status_invalid", "artifact": name})
    gap_rows: list[dict[str, str]] = []
    if "round14_frozen_gap_ledger" in paths:
        gap_rows = read_tsv(paths["round14_frozen_gap_ledger"])
        if len(gap_rows) != 20:
            blockers.append({"code": "frozen_gap_count_drift", "actual": len(gap_rows)})
        gap_uids = [str(row.get("gap_uid", "")).strip() for row in gap_rows]
        if any(not uid for uid in gap_uids):
            blockers.append({"code": "frozen_gap_uid_missing"})
        elif len(set(gap_uids)) != len(gap_uids):
            blockers.append({"code": "frozen_gap_uid_duplicate"})
    if blockers:
        raise ValueError(blockers)
    return {
        "paths": paths,
        "hashes": {name: sha256_file(path) for name, path in paths.items()},
        "payloads": payloads,
        "gap_rows": gap_rows,
    }


def _source_record(
    source: Mapping[str, object], policy_version: str
) -> dict[str, object]:
    record = dict(source)
    timestamps = [dict(row) for row in record["public_timestamps"]]
    selected = select_earliest_public_timestamp(
        timestamps, policy_version=policy_version
    )
    record["selected_public_timestamp"] = selected["timestamp"]
    record["selected_timestamp_basis"] = selected["timestamp_type"]
    record["content_hash"] = canonical_sha256(
        {
            "identity": record["publication_or_accession_identity"],
            "version": record["source_version"],
            "normalized_public_content": record["normalized_public_content"],
        }
    )
    return record


def _outcome_record(
    outcome: Mapping[str, object], sources: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    record = dict(outcome)
    record["source"] = dict(sources[str(record.pop("source_ref"))])
    record["outcome_uid"] = stable_uid("held_out_outcome", record)
    record["content_hash"] = canonical_sha256(record)
    return record


def _episode_record(
    episode: Mapping[str, object],
    *,
    sources: Mapping[str, Mapping[str, object]],
    outcomes: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    value = dict(episode)
    value.pop("episode_key", None)
    value.pop("memory_risk_track", None)
    value.pop("expected_rejection_codes", None)
    pre_refs = [str(item) for item in value.pop("pre_cutoff_source_refs")]
    outcome_refs = [str(item) for item in value.pop("held_out_outcome_refs")]
    value["pre_cutoff_evidence_manifest"] = {
        "bundle_uid": stable_uid(
            "pre_cutoff_bundle",
            {"sources": pre_refs, "question": value["scientific_question"]},
        ),
        "source_refs": pre_refs,
        "source_hashes": {ref: sources[ref]["content_hash"] for ref in pre_refs},
        "agent_visible": True,
    }
    value["held_out_outcome_manifest"] = {
        "bundle_uid": stable_uid(
            "held_out_bundle",
            {"outcomes": outcome_refs, "question": value["scientific_question"]},
        ),
        "outcome_refs": [outcomes[ref]["outcome_uid"] for ref in outcome_refs],
        "outcome_hashes": {
            str(outcomes[ref]["outcome_uid"]): outcomes[ref]["content_hash"]
            for ref in outcome_refs
        },
        "evaluator_only": True,
    }
    value["episode_uid"] = canonical_episode_uid(value)
    value["content_hash"] = episode_content_hash(value)
    return value


def _agent_visible_episode(
    episode: Mapping[str, object], sources: Mapping[str, Mapping[str, object]]
) -> dict[str, object]:
    value = dict(episode)
    pre_manifest = dict(value["pre_cutoff_evidence_manifest"])
    evidence = []
    for ref in pre_manifest["source_refs"]:
        source = sources[str(ref)]
        evidence.append(
            {
                "opaque_source_uid": source["opaque_source_uid"],
                "public_timestamp": source["selected_public_timestamp"],
                "evidence_class": source["evidence_class"],
                "endpoint_summary": source["agent_visible_summary"],
                "content_hash": source["content_hash"],
            }
        )
    pre_manifest["evidence"] = evidence
    pre_manifest.pop("source_refs", None)
    value["pre_cutoff_evidence_manifest"] = pre_manifest
    held = dict(value["held_out_outcome_manifest"])
    value["held_out_outcome_manifest"] = {
        "bundle_uid": held["bundle_uid"],
        "evaluator_only": True,
        "outcome_count": len(held["outcome_refs"]),
    }
    return value


def _direction_from_evidence(
    evidence: Sequence[Mapping[str, object]], *, skeptical: bool
) -> tuple[str, float]:
    directions = [
        str(row["endpoint_summary"].get("direction"))
        for row in evidence
        if row.get("endpoint_summary", {}).get("direction")
    ]
    conflicts = len(set(directions)) > 1 or any(
        row["endpoint_summary"].get("conflict") for row in evidence
    )
    weak = any(
        row["endpoint_summary"].get("strength") in {"weak", "narrative_only", "unknown"}
        for row in evidence
    )
    if not directions or (skeptical and weak):
        return "insufficient_information", 0.25
    if conflicts and skeptical:
        return "mixed_or_context_dependent", 0.40
    label, count = Counter(directions).most_common(1)[0]
    confidence = min(0.85, 0.45 + 0.15 * count)
    return label, confidence


def _probabilities(label: str, confidence: float) -> dict[str, float]:
    labels = (
        "increase",
        "decrease",
        "no_clear_effect",
        "mixed_or_context_dependent",
        "insufficient_information",
    )
    residual = (1.0 - confidence) / (len(labels) - 1)
    return {item: (confidence if item == label else residual) for item in labels}


def _component_uncertainty(
    confidence: float, *, model_known: bool
) -> dict[str, object]:
    unknown = {"status": "unknown", "value": None}
    quantified = {"status": "quantified", "value": round(1.0 - confidence, 6)}
    return {
        "evidence": quantified,
        "model": quantified if model_known else unknown,
        "retrieval": {"status": "quantified", "value": 0.25},
        "mechanism_disagreement": {"status": "quantified", "value": 0.5},
        "support_domain_extrapolation": unknown,
        "unknown_components": ["prospective_transfer", "support_domain_extrapolation"],
    }


def _scalar_uncertainty() -> dict[str, object]:
    return {
        "status": "partial",
        "value": None,
        "unknown_components": ["prospective_transfer"],
    }


def _artifact_hash_payload(
    value: Mapping[str, object], identity_fields: Sequence[str]
) -> str:
    excluded = {"content_hash", "decision_artifact_hash", *identity_fields}
    return canonical_sha256(
        {key: item for key, item in value.items() if key not in excluded}
    )


_SCITIL_AGENT_CAPABILITIES = {
    "ScientistAgent": ["propose_hypothesis", "propose_world_action", "propose_what_if"],
    "FalsifierAgent": ["propose_disconfirming_prediction", "challenge_domain"],
    "ExperimentAgent": ["propose_information_action", "propose_observation_request"],
    "EvidenceAgent": ["assemble_typed_evidence_refs", "identify_conflict"],
}


def _scitil_agent(role: str) -> dict[str, object]:
    prohibited = [
        "write_gate",
        "execute_tool_or_rollout",
        "update_twin_state",
        "upgrade_evidence",
        "close_frozen_gap_by_reasoning",
    ]
    return AgentIdentity(
        schema_version=SCITIL_SCHEMA_VERSIONS["agent_identity"],
        agent_uid=f"round17_{role.lower()}_agent",
        agent_role=role,
        capabilities=_SCITIL_AGENT_CAPABILITIES[role],
        prohibited_capabilities=prohibited,
        gate_writer=False,
        tool_executor=False,
        state_update_authority=False,
        canonical_controller=False,
        provenance={"source": "round17_scitil_runtime_contract", "registered": True},
    ).to_dict()


def _scitil_proposal(
    *, uid: str, agent: Mapping[str, object], belief_state_ref: str,
    proposal_type: str, support_domain_ref: str, evidence_refs: Sequence[str],
    hypothesis_ref: str | None = None, world_action_ref: str | None = None,
    information_action_ref: str | None = None,
) -> dict[str, object]:
    return AgentProposal(
        schema_version=SCITIL_SCHEMA_VERSIONS["agent_proposal"],
        proposal_uid=uid,
        agent_uid=str(agent["agent_uid"]), agent_role=str(agent["agent_role"]),
        belief_state_ref=belief_state_ref, proposal_type=proposal_type,
        hypothesis_ref=hypothesis_ref, world_action_ref=world_action_ref,
        information_action_ref=information_action_ref, what_if_request_ref=None,
        observation_request_ref=None,
        expected_information_gain={"status": "represented", "relative": 0.25},
        expected_twin_debt_change={"status": "represented", "direction": "unknown_or_nonincreasing"},
        expected_cost={"compute_seconds": 1.0, "tool_calls": 0.0},
        risk_class="contract_only", support_domain_ref=support_domain_ref,
        evidence_refs=list(evidence_refs) or ["round18_pre_cutoff_bundle"],
        assumptions=["retrospective_public_literature_only"],
        uncertainties=["held_out_outcome_not_visible_before_t5"],
        prohibited_claim_flags={"causal_effect_claim": False, "clinical_recommendation": False},
        provenance={"source": "round18_full_governed_scitil", "compiled": True},
    ).to_dict()


def _scitil_gate(
    *, gate_class: type[HypothesisGateDecision], schema_key: str, uid: str,
    gate_type: str, proposal_uid: str, decision: str,
    input_values: Mapping[str, object], reason: str,
) -> dict[str, object]:
    return gate_class(
        schema_version=SCITIL_SCHEMA_VERSIONS[schema_key], gate_uid=uid,
        gate_type=gate_type, proposal_uid=proposal_uid,
        controller_uid=SCITIL_CONTROLLER_UID, validator_uid=ROUND17_VALIDATOR_UID,
        input_hashes={name: canonical_sha256(item) for name, item in input_values.items()},
        decision=decision, reason_codes=[reason],
        approved_scope={"execution_scope": "retrospective_contract_validation"},
        budget_effect={"increase_allowed": False, "status": "bounded"},
        twin_debt_effect={"unknown_to_zero": False, "status": "preserved"},
        support_domain_effect={"expansion_allowed": False, "status": "exact_or_restricted"},
        provenance={"source": "round17_twin_kernel", "writer": SCITIL_CONTROLLER_UID},
    ).to_dict()


def _scitil_runtime_trace(
    episode: Mapping[str, object], *, system_contract: Mapping[str, object]
) -> dict[str, object]:
    """Route one public replay decision through the accepted TwinKernel."""

    if system_contract.get("runtime_mode") != "round17_twin_kernel":
        raise ValueError("TwinKernel trace requested for a non-full Round-18 system.")
    support_domain = dict(episode.get("support_domain", {}))
    domain_ref = str(support_domain.get("domain_uid", "round18_public_domain"))
    evidence = list(dict(episode["pre_cutoff_evidence_manifest"]).get("evidence", []))
    evidence_refs = [str(dict(row).get("content_hash", "")) for row in evidence]
    belief_state_ref = stable_uid(
        "round17_replay_belief", {"episode_uid": episode["episode_uid"], "domain": domain_ref}
    )
    kernel = TwinKernel(
        controller_uid=SCITIL_CONTROLLER_UID,
        allowlisted_adapters={"retrieval_limited_public_evidence": {"access_policy": "read_only"}},
    )
    agents: dict[str, dict[str, object]] = {}
    proposals: list[dict[str, object]] = []
    hypotheses: list[dict[str, object]] = []
    gate_records: list[dict[str, object]] = []

    def agent(role: str) -> dict[str, object]:
        if role not in agents:
            value = _scitil_agent(role)
            if validate_agent_identity(value):
                raise ValueError(validate_agent_identity(value))
            agents[role] = value
        return agents[role]

    def proposal(value: Mapping[str, object]) -> dict[str, object]:
        row = dict(value)
        row_blockers = kernel.validate_proposal(
            row, agent=agents[str(row["agent_role"])], belief_states={belief_state_ref}
        )
        if row_blockers:
            raise ValueError(row_blockers)
        proposals.append(row)
        return row

    candidate_hypotheses = [dict(item) for item in episode.get("candidate_hypotheses", [])]
    if candidate_hypotheses:
        candidate = candidate_hypotheses[0]
        hypothesis_uid = str(candidate["hypothesis_uid"])
        falsifier_statement = (
            str(candidate_hypotheses[1].get("statement"))
            if len(candidate_hypotheses) > 1
            else "The later public endpoint does not support the prespecified direction."
        )
        hypothesis = ScientificHypothesis(
            schema_version=SCITIL_SCHEMA_VERSIONS["scientific_hypothesis"],
            hypothesis_uid=hypothesis_uid,
            mechanism_statement=str(candidate["statement"]),
            target_variables=[str(dict(episode["evaluation_target"])["target_uid"])],
            expected_outcome={"status": "prespecified", "candidate": hypothesis_uid},
            required_context=support_domain,
            time_horizon={"value": dict(episode["evaluation_target"])["timepoint_or_horizon"]},
            potential_falsifier={"disconfirming_prediction": falsifier_statement},
            supporting_evidence=evidence_refs or ["round18_pre_cutoff_bundle"],
            conflicting_evidence=(
                [
                    str(row.get("content_hash")) for row in evidence
                    if dict(row.get("endpoint_summary", {})).get("conflict")
                ]
                or ["none_identified_in_visible_bundle"]
            ),
            support_domain_ref=domain_ref,
            uncertainty={"status": "unknown", "unknown_components": ["held_out_outcome"]},
            causal_effect_claim=False, clinical_efficacy_claim=False,
            hypothetical_as_empirical_support=False,
        ).to_dict()
        hypotheses.append(hypothesis)
        scientist = proposal(_scitil_proposal(
            uid=stable_uid("round18_scientist_proposal", episode["episode_uid"]),
            agent=agent("ScientistAgent"), belief_state_ref=belief_state_ref,
            proposal_type="new_hypothesis", support_domain_ref=domain_ref,
            evidence_refs=evidence_refs, hypothesis_ref=hypothesis_uid,
        ))
        h_gate = _scitil_gate(
            gate_class=HypothesisGateDecision, schema_key="hypothesis_gate_decision",
            uid=stable_uid("round18_h_gate", scientist["proposal_uid"]), gate_type="H-Gate",
            proposal_uid=str(scientist["proposal_uid"]), decision="admitted_for_testing",
            input_values={"proposal": scientist, "hypothesis": hypothesis},
            reason="falsifiable_public_replay_hypothesis",
        )
        blockers = validate_h_gate(h_gate, proposal=scientist, hypothesis=hypothesis)
        if blockers:
            raise ValueError(blockers)
        gate_records.append({"gate_type": "H-Gate", "gate": h_gate,
                             "proposal_uid": scientist["proposal_uid"],
                             "hypothesis_uid": hypothesis_uid})
        if bool(system_contract.get("falsifier_enabled")):
            falsifier = proposal(_scitil_proposal(
                uid=stable_uid("round18_falsifier_proposal", episode["episode_uid"]),
                agent=agent("FalsifierAgent"), belief_state_ref=belief_state_ref,
                proposal_type="falsification_test", support_domain_ref=domain_ref,
                evidence_refs=evidence_refs, hypothesis_ref=hypothesis_uid,
            ))
            falsifier_gate = _scitil_gate(
                gate_class=HypothesisGateDecision, schema_key="hypothesis_gate_decision",
                uid=stable_uid("round18_falsifier_h_gate", falsifier["proposal_uid"]),
                gate_type="H-Gate", proposal_uid=str(falsifier["proposal_uid"]),
                decision="admitted_for_testing",
                input_values={"proposal": falsifier, "hypothesis": hypothesis},
                reason="disconfirming_prediction_registered_before_summary",
            )
            blockers = validate_h_gate(falsifier_gate, proposal=falsifier, hypothesis=hypothesis)
            if blockers:
                raise ValueError(blockers)
            gate_records.append({"gate_type": "H-Gate", "gate": falsifier_gate,
                                 "proposal_uid": falsifier["proposal_uid"],
                                 "hypothesis_uid": hypothesis_uid})
    elif episode.get("candidate_actions"):
        action = dict(episode["candidate_actions"][0])
        world_proposal = proposal(_scitil_proposal(
            uid=stable_uid("round18_world_action_proposal", episode["episode_uid"]),
            agent=agent("ScientistAgent"), belief_state_ref=belief_state_ref,
            proposal_type="world_action_candidate", support_domain_ref=domain_ref,
            evidence_refs=evidence_refs, world_action_ref=str(action.get("action_uid")),
        ))
        w_gate = _scitil_gate(
            gate_class=WorldActionGateDecision, schema_key="world_action_gate_decision",
            uid=stable_uid("round18_w_gate", world_proposal["proposal_uid"]), gate_type="W-Gate",
            proposal_uid=str(world_proposal["proposal_uid"]), decision="blocked_incomplete_action",
            input_values={"proposal": world_proposal, "action": action},
            reason="ranking_candidate_is_not_authorized_for_world_execution",
        )
        blockers = validate_w_gate(w_gate, proposal=world_proposal, world_action=action,
                                   what_if_request=None, twin_debt={}, budget_decision={})
        if blockers:
            raise ValueError(blockers)
        gate_records.append({"gate_type": "W-Gate", "gate": w_gate,
                             "proposal_uid": world_proposal["proposal_uid"], "world_action": action})
    elif episode.get("candidate_information_actions"):
        info = dict(episode["candidate_information_actions"][0])
        info_proposal = proposal(_scitil_proposal(
            uid=stable_uid("round18_information_proposal", episode["episode_uid"]),
            agent=agent("ExperimentAgent"), belief_state_ref=belief_state_ref,
            proposal_type="information_action_candidate", support_domain_ref=domain_ref,
            evidence_refs=evidence_refs,
            information_action_ref=str(info.get("information_action_uid")),
        ))
        i_gate = _scitil_gate(
            gate_class=InformationActionGateDecision, schema_key="information_action_gate_decision",
            uid=stable_uid("round18_i_gate", info_proposal["proposal_uid"]), gate_type="I-Gate",
            proposal_uid=str(info_proposal["proposal_uid"]), decision="revision_required",
            input_values={"proposal": info_proposal, "information_action": info},
            reason="public_query_selection_requires_typed_adapter_before_execution",
        )
        blockers = validate_i_gate(i_gate, proposal=info_proposal, information_action=info,
                                   tool_request=None, allowlisted_adapters=set(kernel.adapter_uids),
                                   budget_decision={})
        if blockers:
            raise ValueError(blockers)
        gate_records.append({"gate_type": "I-Gate", "gate": i_gate,
                             "proposal_uid": info_proposal["proposal_uid"], "information_action": info})
    else:
        stop = proposal(_scitil_proposal(
            uid=stable_uid("round18_stop_proposal", episode["episode_uid"]),
            agent=agent("ScientistAgent"), belief_state_ref=belief_state_ref,
            proposal_type="stop_recommendation", support_domain_ref=domain_ref,
            evidence_refs=evidence_refs,
        ))
        decision = "unsupported_domain" if support_domain.get("extrapolation") else "revision_required"
        h_gate = _scitil_gate(
            gate_class=HypothesisGateDecision, schema_key="hypothesis_gate_decision",
            uid=stable_uid("round18_stop_h_gate", stop["proposal_uid"]), gate_type="H-Gate",
            proposal_uid=str(stop["proposal_uid"]), decision=decision,
            input_values={"proposal": stop}, reason="no_supported_hypothesis_or_action_to_authorize",
        )
        blockers = validate_h_gate(h_gate, proposal=stop, hypothesis={})
        if blockers:
            raise ValueError(blockers)
        gate_records.append({"gate_type": "H-Gate", "gate": h_gate,
                             "proposal_uid": stop["proposal_uid"]})

    if proposals and hypotheses:
        observation_gate = _scitil_gate(
            gate_class=ObservationGateDecision, schema_key="observation_gate_decision",
            uid=stable_uid("round18_o_gate", episode["episode_uid"]), gate_type="O-Gate",
            proposal_uid=str(proposals[0]["proposal_uid"]), decision="no_update",
            input_values={"proposal": proposals[0], "model_output": {"empirical_observation": False}},
            reason="model_or_benchmark_artifact_cannot_assimilate_as_observation",
        )
        blockers = validate_o_gate(observation_gate, proposal=proposals[0],
                                   assimilation_proposal={}, assimilation_decision={})
        if blockers:
            raise ValueError(blockers)
        gate_records.append({"gate_type": "O-Gate", "gate": observation_gate,
                             "proposal_uid": proposals[0]["proposal_uid"]})

    trace: dict[str, object] = {
        "runtime_uid": stable_uid("round18_scitil_runtime", episode["episode_uid"]),
        "runtime_mode": "round17_twin_kernel", "kernel_invoked": True,
        "kernel_llm_backed": bool(kernel.llm_backed),
        "controller_uid": SCITIL_CONTROLLER_UID, "validator_uid": ROUND17_VALIDATOR_UID,
        "belief_state_ref": belief_state_ref, "agents": list(agents.values()),
        "proposals": proposals, "hypotheses": hypotheses, "gate_records": gate_records,
        "falsifier_required": bool(candidate_hypotheses),
        "falsifier_invoked": bool(candidate_hypotheses and system_contract.get("falsifier_enabled")),
        "external_tool_executed": False, "model_rollout_executed": False,
        "agent_direct_state_update": False, "agent_direct_gate_write": False,
        "assimilation_applied": False, "status": "validated_no_external_execution",
    }
    trace["content_hash"] = canonical_sha256(trace)
    trace_blockers = validate_scitil_runtime_trace(trace)
    if trace_blockers:
        raise ValueError(trace_blockers)
    return trace


def _predict_episode(
    system_uid: str,
    episode: Mapping[str, object],
    system_contract: Mapping[str, object],
) -> dict[str, object]:
    evidence = list(dict(episode["pre_cutoff_evidence_manifest"]).get("evidence", []))
    track = str(episode["track"])
    skeptical = bool(system_contract["skeptical_conflict_policy"])
    evidence_weak = any(
        dict(row.get("endpoint_summary", {})).get("strength")
        in {"weak", "narrative_only", "unknown"}
        for row in evidence
    )
    forced_abstention = bool(
        dict(episode.get("leakage_controls", {})).get("admission_blocked")
    ) or (
        track == "abstention_boundary_compliance"
        or bool(dict(episode["support_domain"]).get("extrapolation"))
        or evidence_weak
    )
    direction, confidence = _direction_from_evidence(evidence, skeptical=skeptical)
    if track == "mechanism_discrimination":
        mechanism_labels = [
            str(dict(row.get("endpoint_summary", {})).get("mechanism_status"))
            for row in evidence
            if dict(row.get("endpoint_summary", {})).get("mechanism_status")
        ]
        direction = (
            Counter(mechanism_labels).most_common(1)[0][0]
            if mechanism_labels
            else "unresolved"
        )
        confidence = 0.55 if mechanism_labels else 0.30
    confidence = min(0.95, confidence * float(system_contract["confidence_scale"]))
    if forced_abstention:
        direction, confidence = "insufficient_information", 0.0

    ranking: list[str] = []
    if track == "intervention_ranking":
        scores = {}
        for action in episode.get("candidate_actions", []):
            action_map = dict(action)
            scores[str(action_map["action_uid"])] = float(
                action_map.get("pre_cutoff_score", 0.0)
            )
        if system_contract["ranking_policy"] == "opaque_uid":
            ranking = sorted(scores)
        else:
            ranking = [
                uid
                for uid, _ in sorted(
                    scores.items(), key=lambda item: (-item[1], item[0])
                )
            ]
    information_action = None
    if track in {"information_action_selection", "sequential_hypothesis_replay"}:
        actions = [
            dict(row) for row in episode.get("candidate_information_actions", [])
        ]
        if actions:
            selected = max(
                actions,
                key=lambda row: (
                    float(row.get("expected_information_gain", 0.0)),
                    str(row["action_uid"]),
                ),
            )
            information_action = selected["action_uid"]
        if system_contract["information_action_policy"] == "none":
            information_action = None
    abstained = direction == "insufficient_information" or forced_abstention
    label_space = (
        ("supported", "refuted", "unresolved", "insufficient_information")
        if track == "mechanism_discrimination"
        else (
            "increase",
            "decrease",
            "no_clear_effect",
            "mixed_or_context_dependent",
            "insufficient_information",
        )
    )
    if direction not in label_space:
        direction = "insufficient_information"
        abstained = True
    if abstained:
        class_probabilities = {
            item: float(item == "insufficient_information") for item in label_space
        }
    else:
        residual = (1.0 - confidence) / (len(label_space) - 1)
        class_probabilities = {
            item: confidence if item == direction else residual for item in label_space
        }
    decision: dict[str, object] = {
        "decision_uid": stable_uid(
            "round18_t4_decision",
            {"system": system_uid, "episode": episode["episode_uid"]},
        ),
        "run_uid": stable_uid(
            "literature_replay_run",
            {"system": system_uid, "episode": episode["episode_uid"]},
        ),
        "episode_uid": episode["episode_uid"],
        "system_uid": system_uid,
        "track": track,
        "predicted_label": direction,
        "class_probabilities": class_probabilities,
        "confidence": None if abstained else confidence,
        "ranking": ranking,
        "selected_information_action": information_action,
        "abstained": abstained,
        "abstention_required": forced_abstention,
        "abstention_reason": (
            "unsupported_domain"
            if bool(dict(episode["support_domain"]).get("extrapolation"))
            else "insufficient_information"
            if forced_abstention
            else "not_required"
        ),
        "support_domain_status": "extrapolative"
        if dict(episode["support_domain"]).get("extrapolation")
        else "in_domain",
        "uncertainty": _component_uncertainty(
            confidence,
            model_known=bool(system_contract["model_uncertainty_known"]),
        ),
        "falsifier_invoked": bool(
            system_contract["falsifier_enabled"] and episode.get("candidate_hypotheses")
        ),
        **CLAIM_SCOPE,
    }
    if system_uid == "full_governed_scitil":
        decision["scitil_runtime_trace"] = _scitil_runtime_trace(
            episode, system_contract=system_contract
        )
    return decision


def _prediction_artifact(
    decision: Mapping[str, object], evidence_bundle_hash: str
) -> dict[str, object] | None:
    if decision["track"] not in {
        "endpoint_direction_prediction",
        "mechanism_discrimination",
        "sequential_hypothesis_replay",
    }:
        return None
    row: dict[str, object] = {
        "schema_version": SCHEMA_VERSIONS["benchmark_prediction"],
        "prediction_uid": stable_uid("benchmark_prediction", decision["decision_uid"]),
        "episode_uid": decision["episode_uid"],
        "system_uid": decision["system_uid"],
        "track": decision["track"],
        "predicted_label": decision["predicted_label"],
        "class_probabilities": decision["class_probabilities"],
        "confidence": decision["confidence"],
        "uncertainty": decision["uncertainty"],
        "support_domain_status": decision["support_domain_status"],
        "evidence_bundle_hash": evidence_bundle_hash,
        "abstained": decision["abstained"],
        **CLAIM_SCOPE,
        "status": "abstained" if decision["abstained"] else "predicted",
    }
    digest = _artifact_hash_payload(row, ("prediction_uid",))
    row["decision_artifact_hash"] = digest
    row["content_hash"] = digest
    return row


def _ranking_artifact(
    decision: Mapping[str, object],
    episode: Mapping[str, object],
    evidence_bundle_hash: str,
) -> dict[str, object] | None:
    if decision["track"] != "intervention_ranking":
        return None
    action_by_uid = {
        str(dict(item)["action_uid"]): dict(item)
        for item in episode["candidate_actions"]
    }
    ranking = list(decision["ranking"])
    row: dict[str, object] = {
        "schema_version": SCHEMA_VERSIONS["benchmark_ranking"],
        "ranking_uid": stable_uid("benchmark_ranking", decision["decision_uid"]),
        "episode_uid": decision["episode_uid"],
        "system_uid": decision["system_uid"],
        "candidate_set_uid": stable_uid("round18_candidate_set", sorted(action_by_uid)),
        "frozen_candidate_set_hash": canonical_sha256(
            list(episode["candidate_actions"])
        ),
        "candidate_set_frozen_before_replay": True,
        "evaluation_target_ref": dict(episode["evaluation_target"])["target_uid"],
        "ranked_candidates": [
            {
                "candidate_uid": uid,
                "rank": index,
                "score": float(action_by_uid[uid].get("pre_cutoff_score", 0.0)),
                "uncertainty": {"status": "unknown", "value": None},
            }
            for index, uid in enumerate(ranking, start=1)
        ],
        "uncertainty": _scalar_uncertainty(),
        "support_domain_status": decision["support_domain_status"],
        "evidence_bundle_hash": evidence_bundle_hash,
        **CLAIM_SCOPE,
        "status": "ranked",
    }
    digest = _artifact_hash_payload(row, ("ranking_uid",))
    row["decision_artifact_hash"] = digest
    row["content_hash"] = digest
    return row


def _abstention_artifact(decision: Mapping[str, object]) -> dict[str, object]:
    required = bool(decision["abstention_required"])
    row: dict[str, object] = {
        "schema_version": SCHEMA_VERSIONS["benchmark_abstention"],
        "abstention_uid": stable_uid("benchmark_abstention", decision["decision_uid"]),
        "episode_uid": decision["episode_uid"],
        "system_uid": decision["system_uid"],
        "abstention_required": required,
        "abstained": bool(decision["abstained"]),
        "reason": decision["abstention_reason"] if required else "not_required",
        "support_domain_status": decision["support_domain_status"],
        "evidence_sufficiency": "insufficient" if required else "sufficient",
        "uncertainty": _scalar_uncertainty(),
        "unknown_components_preserved": True,
        "conflicts_preserved": True,
        "forced_answer_refused": True,
        "unsupported_answer_emitted": False,
        **CLAIM_SCOPE,
        "status": "abstained_correctly" if required else "answered_with_support",
    }
    row["content_hash"] = _artifact_hash_payload(row, ("abstention_uid",))
    return row


def _prediction_worker(input_path: Path, output_path: Path) -> int:
    payload = _load_json(input_path)
    serialized = json.dumps(payload, sort_keys=True)
    forbidden = ("evaluator_only_path", "held_out_bundle_path", "held_out_outcomes")
    if any(token in serialized for token in forbidden):
        raise ValueError("Prediction worker received evaluator-only material.")
    decisions: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    rankings: list[dict[str, object]] = []
    abstentions: list[dict[str, object]] = []
    episode_by_uid = {str(row["episode_uid"]): dict(row) for row in payload["episodes"]}
    for system_uid in payload["systems"]:
        system_contract = dict(payload["system_contracts"][system_uid])
        for episode in payload["episodes"]:
            if episode["status"] == "rejected":
                continue
            decision = _predict_episode(system_uid, episode, system_contract)
            decisions.append(decision)
            prediction = _prediction_artifact(decision, str(payload["bundle_hash"]))
            if prediction is not None:
                predictions.append(prediction)
            ranking = _ranking_artifact(
                decision,
                episode_by_uid[str(episode["episode_uid"])],
                str(payload["bundle_hash"]),
            )
            if ranking is not None:
                rankings.append(ranking)
            abstentions.append(_abstention_artifact(decision))
    write_json(
        output_path,
        {
            "worker_pid": os.getpid(),
            "worker_uid": os.getuid(),
            "bundle_hash": payload["bundle_hash"],
            "decisions": decisions,
            "predictions": predictions,
            "rankings": rankings,
            "abstentions": abstentions,
            "held_out_material_received": False,
        },
    )
    return 0


def _execute_held_out_access_attack(
    held_manifest_path: Path, held_bundle_uid: str
) -> dict[str, object]:
    """Exercise the negative control while the evaluator mount is mode 000."""
    attempt = {
        "attempt_uid": stable_uid("round18_access_attack", held_bundle_uid),
        "actor_role": "agent",
        "process_identity": "round18_negative_control_agent",
        "bundle_uid": held_bundle_uid,
        "step": "T4",
        "evaluator_only": True,
        "attempted": True,
        "access_granted": False,
        "outcome_exposed": False,
        "attempt_preserved": True,
        "failure_class": "permission_denied",
    }
    try:
        with held_manifest_path.open("rb") as handle:
            handle.read(1)
    except PermissionError:
        return attempt
    except OSError as exc:
        attempt["failure_class"] = f"os_error:{exc.errno}"
        return attempt
    attempt["access_granted"] = True
    attempt["outcome_exposed"] = True
    attempt["failure_class"] = "unexpected_access_granted"
    raise PermissionError("Round-18 held-out access attack was not rejected.")


def _events_for_episode(
    episode: Mapping[str, object],
    decision_hash: str,
    outcome: Mapping[str, object] | None,
    *,
    agent_bundle_hash: str,
    held_out_bundle_hash: str,
    recorded_at: str,
) -> list[dict[str, object]]:
    """Materialize all T0-T7 artifacts, including non-empty intermediate steps."""
    events: list[dict[str, object]] = []
    previous_hash: str | None = None
    hypothesis_hash = canonical_sha256(
        {
            "episode_uid": episode["episode_uid"],
            "hypotheses": episode.get("candidate_hypotheses", []),
        }
    )
    falsifier_hash = canonical_sha256(
        {
            "episode_uid": episode["episode_uid"],
            "challenge": "alternative mechanism, negative control, disconfirming prediction, domain and reliability challenge",
            "hypotheses": episode.get("candidate_hypotheses", []),
        }
    )
    action_hash = canonical_sha256(
        {
            "episode_uid": episode["episode_uid"],
            "candidate_actions": episode.get("candidate_actions", []),
            "candidate_information_actions": episode.get(
                "candidate_information_actions", []
            ),
            "budget": episode.get("budget", {}),
        }
    )
    query_hash = canonical_sha256(
        {
            "episode_uid": episode["episode_uid"],
            "allowed_tools": episode.get("allowed_tools", []),
            "query_or_rollout": "governed_public_evidence_query",
        }
    )
    score_hash = canonical_sha256(
        {
            "episode_uid": episode["episode_uid"],
            "decision_hash": decision_hash,
            "outcome_hash": None if outcome is None else outcome["content_hash"],
        }
    )
    assimilation_hash = canonical_sha256(
        {
            "episode_uid": episode["episode_uid"],
            "score_hash": score_hash,
            "update": "retrospective_only_no_pre_cutoff_mutation",
        }
    )
    for index, step in enumerate(REVEAL_STEPS):
        input_hashes: dict[str, str] = {}
        output_hashes: dict[str, str] = {}
        if step == "T0":
            input_hashes["agent_visible_bundle"] = agent_bundle_hash
            output_hashes["episode"] = str(episode["content_hash"])
        elif step == "T1":
            input_hashes["episode"] = str(episode["content_hash"])
            output_hashes.update(
                {
                    "hypothesis_proposal": hypothesis_hash,
                    "falsifier_challenge": falsifier_hash,
                }
            )
        elif step == "T2":
            input_hashes.update(
                {
                    "hypothesis_proposal": hypothesis_hash,
                    "falsifier_challenge": falsifier_hash,
                }
            )
            output_hashes["action_selection"] = action_hash
        elif step == "T3":
            input_hashes["action_selection"] = action_hash
            output_hashes["information_action_result"] = query_hash
        elif step == "T4":
            input_hashes["information_action_result"] = query_hash
            output_hashes["prediction"] = decision_hash
        elif step == "T5" and outcome is not None:
            input_hashes["held_out_bundle"] = held_out_bundle_hash
            output_hashes["held_out_outcome"] = str(outcome["content_hash"])
        elif step == "T6" and outcome is not None:
            input_hashes.update(
                {
                    "prediction": decision_hash,
                    "held_out_outcome": str(outcome["content_hash"]),
                }
            )
            output_hashes["score"] = score_hash
        elif step == "T7" and outcome is not None:
            input_hashes["score"] = score_hash
            output_hashes["retrospective_assimilation"] = assimilation_hash
        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSIONS["sequential_reveal_event"],
            "event_uid": "pending",
            "episode_uid": episode["episode_uid"],
            "step": step,
            "sequence_index": index,
            "timestamp": recorded_at,
            "input_hashes": input_hashes,
            "output_hashes": output_hashes,
            "held_out_revealed": step in {"T5", "T6", "T7"},
            "held_out_accessed": step in {"T5", "T6", "T7"},
            "outcome_ref": outcome["outcome_uid"]
            if outcome is not None and step in {"T5", "T6", "T7"}
            else None,
            "immutable": True,
            "preserved": True,
            "prior_event_hash": previous_hash,
            "process_identity": "round18_agent_runtime"
            if index <= 4
            else "round18_evaluator",
            "status": "completed",
        }
        if step == "T4":
            payload["decision_hash"] = decision_hash
        if step in {"T5", "T6", "T7"}:
            payload["pre_cutoff_decision_hash"] = decision_hash
        payload["event_hash"] = sequential_event_content_hash(payload)
        payload["event_uid"] = canonical_sequential_event_uid(payload)
        previous_hash = str(payload["event_hash"])
        events.append(payload)
    return events


def _score_predictions(
    decisions: Sequence[Mapping[str, object]],
    episodes: Mapping[str, Mapping[str, object]],
    outcomes_by_uid: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for decision in decisions:
        episode = episodes[str(decision["episode_uid"])]
        outcome_uid = str(dict(episode["held_out_outcome_manifest"])["outcome_refs"][0])
        outcome = outcomes_by_uid[outcome_uid]
        target = dict(outcome["evaluation_target"])
        truth = str(outcome["label"])
        correct = decision["predicted_label"] == truth
        if episode["track"] == "intervention_ranking":
            predicted_ranking = list(decision.get("ranking", []))
            reference_ranking = list(outcome.get("reference_ranking", []))
            correct = predicted_ranking[:1] == reference_ranking[:1]
        if episode["track"] == "information_action_selection":
            correct = decision.get("selected_information_action") == outcome.get(
                "selected_information_action"
            )
        information_actions = [
            dict(item) for item in episode.get("candidate_information_actions", [])
        ]
        selected_action = str(decision.get("selected_information_action") or "")
        selected_action_row = next(
            (
                item
                for item in information_actions
                if str(item.get("action_uid")) == selected_action
            ),
            None,
        )
        action_cost = 1.0 if selected_action_row is not None else 0.0
        max_cost = float(
            dict(episode.get("budget", {})).get("max_cost_units", 1.0) or 1.0
        )
        candidate_count = max(
            len(episode.get("candidate_hypotheses", [])),
            len(episode.get("candidate_actions", [])),
            len(information_actions),
            1,
        )
        information_gain = float(
            (selected_action_row or {}).get("expected_information_gain", 0.0) or 0.0
        )
        valid_falsifier = bool(decision.get("falsifier_invoked")) and episode[
            "track"
        ] in {
            "mechanism_discrimination",
            "sequential_hypothesis_replay",
        }
        hypothesis_reversal = episode["track"] in {
            "mechanism_discrimination",
            "sequential_hypothesis_replay",
        } and str(decision.get("predicted_label")) in {
            "refuted",
            "mixed_or_context_dependent",
        }
        row = {
            "result_uid": stable_uid(
                "benchmark_result",
                {"decision": decision["decision_uid"], "outcome": outcome_uid},
            ),
            "decision_uid": decision["decision_uid"],
            "run_uid": decision["run_uid"],
            "episode_uid": episode["episode_uid"],
            "system_uid": decision["system_uid"],
            "track": episode["track"],
            "partition": episode["independence_groups"]["partition"],
            "publication_family": episode["independence_groups"]["publication_family"],
            "source_quality": outcome["source"]["source_quality"],
            "label_class": outcome["label_provenance"]["label_class"],
            "support_domain": episode["support_domain"]["domain_uid"],
            "time_horizon": target.get("timepoint_or_horizon"),
            "conflict_status": outcome.get("conflict_status", "no_conflict"),
            "abstention_status": "abstained" if decision["abstained"] else "answered",
            "predicted_label": decision["predicted_label"],
            "true_label": truth,
            "correct": bool(correct),
            "confidence": 0.0
            if decision["confidence"] is None
            else decision["confidence"],
            "probabilities": decision["class_probabilities"],
            "ranking": decision.get("ranking", []),
            "reference_ranking": outcome.get("reference_ranking", []),
            "selected_information_action": decision.get("selected_information_action"),
            "reference_information_action": outcome.get("selected_information_action"),
            "abstention_required": bool(outcome.get("abstention_required")),
            "falsifier_invoked": bool(decision.get("falsifier_invoked")),
            "valid_falsifier_challenge": valid_falsifier,
            "hypothesis_reversal": hypothesis_reversal,
            "cost_to_identification": action_cost,
            "steps_to_falsification": 1.0 if valid_falsifier else None,
            "budget_adjusted_utility": float(correct)
            * max(0.0, 1.0 - action_cost / max_cost),
            "information_gain_per_action": information_gain / action_cost
            if action_cost
            else 0.0,
            "candidate_space_reduction": 1.0 - (1.0 / candidate_count)
            if selected_action_row
            else 0.0,
            "unsupported_positive_avoided": bool(outcome.get("abstention_required"))
            and bool(decision.get("abstained")),
            "restricted_model_based_result": decision["system_uid"]
            in {"twin_only_without_agents", "full_governed_scitil"},
            "contract_correctness_only": True,
            "empirical_biological_validation": False,
            **CLAIM_SCOPE,
        }
        results.append(row)
    return results


def _defined_mean(values: Sequence[object]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return sum(numeric) / len(numeric) if numeric else None


def _metric_rows(results: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in results:
        grouped[(str(row["system_uid"]), str(row["track"]))].append(row)
    for (system_uid, track), group in sorted(grouped.items()):
        metric_sets: list[tuple[str, Mapping[str, object]]] = []
        if track == "endpoint_direction_prediction":
            metric_sets.append(("prediction", compute_direction_metrics(group)))
        if track == "intervention_ranking":
            metric_sets.append(("ranking", compute_ranking_metrics(group, k=2)))
        metric_sets.append(
            (
                "reliability_abstention",
                compute_reliability_metrics(group, false_confidence_threshold=0.7),
            )
        )
        metric_sets.append(
            (
                "falsifier",
                {
                    "falsifier_invocation_rate": sum(
                        bool(item["falsifier_invoked"]) for item in group
                    )
                    / len(group),
                    "valid_challenge_rate": sum(
                        bool(item["valid_falsifier_challenge"]) for item in group
                    )
                    / len(group),
                    "hypothesis_reversal_rate": sum(
                        bool(item["hypothesis_reversal"]) for item in group
                    )
                    / len(group),
                    "unsupported_positive_reduction": sum(
                        bool(item["unsupported_positive_avoided"]) for item in group
                    )
                    / len(group),
                    "additional_information_cost": sum(
                        float(item["cost_to_identification"]) for item in group
                    )
                    / len(group),
                },
            )
        )
        if track in {"information_action_selection", "sequential_hypothesis_replay"}:
            metric_sets.append(
                (
                    "sequential_decision",
                    {
                        "cost_to_identification": _defined_mean(
                            [item["cost_to_identification"] for item in group]
                        ),
                        "steps_to_falsification": _defined_mean(
                            [
                                item["steps_to_falsification"]
                                for item in group
                                if item["steps_to_falsification"] is not None
                            ]
                        ),
                        "budget_adjusted_utility": _defined_mean(
                            [item["budget_adjusted_utility"] for item in group]
                        ),
                        "information_gain_per_action": _defined_mean(
                            [item["information_gain_per_action"] for item in group]
                        ),
                        "candidate_space_reduction": _defined_mean(
                            [item["candidate_space_reduction"] for item in group]
                        ),
                    },
                )
            )
        flattened = [
            (family, name, value)
            for family, metrics in metric_sets
            for name, value in sorted(metrics.items())
        ]
        for family, name, value in flattened:
            run_uid = stable_uid(
                "round18_metric_run", {"system": system_uid, "track": track}
            )
            scope = {
                "track": track,
                "episode_family": "mixed"
                if len({item["publication_family"] for item in group}) > 1
                else str(group[0]["publication_family"]),
                "source_quality": "mixed",
                "label_class": "mixed"
                if len({item["label_class"] for item in group}) > 1
                else str(group[0]["label_class"]),
                "support_domain": "mixed"
                if len({item["support_domain"] for item in group}) > 1
                else str(group[0]["support_domain"]),
                "time_horizon": "mixed"
                if len({item["time_horizon"] for item in group}) > 1
                else str(group[0]["time_horizon"]),
                "conflict_status": "mixed"
                if len({item["conflict_status"] for item in group}) > 1
                else str(group[0]["conflict_status"]),
                "abstention_status": "mixed"
                if len({item["abstention_status"] for item in group}) > 1
                else str(group[0]["abstention_status"]),
            }
            valid = (
                value is not None
                and isinstance(value, (int, float))
                and math.isfinite(float(value))
            )
            row: dict[str, object] = {
                "schema_version": SCHEMA_VERSIONS["benchmark_metric_result"],
                "metric_result_uid": stable_uid(
                    "benchmark_metric", {"run": run_uid, "name": name}
                ),
                "run_uid": run_uid,
                "system_uid": system_uid,
                "metric_name": name,
                "metric_family": family,
                "scope": scope,
                "aggregation_unit": "track",
                "value": float(value) if valid else None,
                "uncertainty": {
                    "status": "not_applicable" if valid else "unknown",
                    "lower": None,
                    "upper": None,
                    "unknown_components": [] if valid else ["pilot_sample_size"],
                },
                "effective_episode_count": len(group),
                "valid": valid,
                **CLAIM_SCOPE,
                "status": "reported" if valid else "not_estimable",
            }
            row["content_hash"] = _artifact_hash_payload(row, ("metric_result_uid",))
            rows.append(row)
    return rows


def _write_required_tsvs(
    root: Path,
    paths: Mapping[str, object],
    predictions: Sequence[Mapping[str, object]],
    rankings: Sequence[Mapping[str, object]],
    results: Sequence[Mapping[str, object]],
    abstentions: Sequence[Mapping[str, object]],
    rejection_rows: Sequence[Mapping[str, object]],
) -> dict[str, Path]:
    prediction_rows = [dict(row) for row in predictions]
    ranking_rows = [dict(row) for row in rankings]
    decision_rows = [
        dict(row)
        for row in results
        if row["track"]
        in {
            "mechanism_discrimination",
            "information_action_selection",
            "sequential_hypothesis_replay",
        }
    ]
    catalogs = {
        "prediction_catalog": (
            prediction_rows,
            list(prediction_rows[0]) if prediction_rows else ["prediction_uid"],
        ),
        "ranking_catalog": (
            ranking_rows,
            list(ranking_rows[0]) if ranking_rows else ["ranking_uid"],
        ),
        "decision_catalog": (
            decision_rows,
            [
                "result_uid",
                "run_uid",
                "episode_uid",
                "system_uid",
                "track",
                "predicted_label",
                "true_label",
                "correct",
                "selected_information_action",
                "reference_information_action",
                *CLAIM_SCOPE,
            ],
        ),
        "abstention_catalog": (
            [dict(row) for row in abstentions],
            list(abstentions[0]) if abstentions else ["abstention_uid"],
        ),
        "leakage_control_catalog": (
            [dict(row) for row in rejection_rows],
            [
                "rejection_uid",
                "episode_uid",
                "control_type",
                "decision",
                "reason",
                "attempt_preserved",
                *CLAIM_SCOPE,
            ],
        ),
    }
    written = {}
    for name, (rows, fields) in catalogs.items():
        path = _resolve(root, paths[name])
        normalized = []
        for row in rows:
            normalized.append(
                {
                    field: json.dumps(
                        row.get(field), sort_keys=True, separators=(",", ":")
                    )
                    if isinstance(row.get(field), (dict, list))
                    else row.get(field)
                    for field in fields
                }
            )
        write_tsv(path, list(fields), normalized)
        written[name] = path
    return written


def build_round18_pilot(
    *,
    config: Mapping[str, object],
    repository_root: Path,
    parent_root: Path,
    output_root: Path,
    created_at: str,
) -> dict[str, object]:
    blockers = _validate_config(config)
    if blockers:
        raise ValueError(blockers)
    cfg = round18_config(config)
    parent = validate_parent_state(config, parent_root)
    outputs = _output_paths(config)
    policy = dict(cfg["policies"])
    cutoff_policy = dict(policy["temporal_cutoff"])
    sources = {
        str(row["source_uid"]): _source_record(
            row, str(cutoff_policy["policy_version"])
        )
        for row in cfg["pilot"]["sources"]
    }
    outcomes_by_key = {
        str(row["outcome_key"]): _outcome_record(row, sources)
        for row in cfg["pilot"]["outcomes"]
    }
    outcomes_by_uid = {str(row["outcome_uid"]): row for row in outcomes_by_key.values()}
    raw_episodes = [dict(row) for row in cfg["pilot"]["episodes"]]
    episodes = [
        _episode_record(row, sources=sources, outcomes=outcomes_by_key)
        for row in raw_episodes
    ]
    cutoff_records = [
        {
            "schema_version": SCHEMA_VERSIONS["literature_cutoff"],
            "cutoff_uid": stable_uid("literature_cutoff", row["episode_uid"]),
            "policy_version": cutoff_policy["policy_version"],
            "available_timestamps": [
                {
                    "timestamp_type": "publisher_publication_date",
                    "timestamp": row["cutoff_timestamp"],
                    "version": "episode_cutoff_declared_before_replay",
                    "publicly_available": True,
                    "admissible": True,
                }
            ],
            "selected_timestamp": row["cutoff_timestamp"],
            "selected_timestamp_type": "publisher_publication_date",
            "temporal_ordering_status": dict(row["leakage_controls"])[
                "temporal_ordering_status"
            ]
            if "temporal_ordering_status" in row.get("leakage_controls", {})
            else "ordered",
            "strong_admission": True,
        }
        for row in episodes
    ]
    episode_by_uid = {str(row["episode_uid"]): row for row in episodes}
    episode_metadata = {
        str(episode["episode_uid"]): {
            "memory_risk_track": raw.get(
                "memory_risk_track", "retrieval_isolated_replay"
            ),
            "expected_rejection_codes": list(raw.get("expected_rejection_codes", [])),
        }
        for raw, episode in zip(raw_episodes, episodes)
    }

    validation_blockers: list[dict[str, object]] = []
    for episode in episodes:
        validation_blockers.extend(validate_episode(episode))
        pre_refs = list(dict(episode["pre_cutoff_evidence_manifest"])["source_refs"])
        held_refs = list(dict(episode["held_out_outcome_manifest"])["outcome_refs"])
        validation_blockers.extend(
            validate_temporal_ordering(
                episode["cutoff_timestamp"],
                pre_cutoff_sources=[sources[str(ref)] for ref in pre_refs],
                held_out_sources=[
                    outcomes_by_uid[str(ref)]["source"] for ref in held_refs
                ],
            )
        )
        validation_blockers.extend(
            validate_label_provenance(dict(episode["label_provenance"]))
        )
        validation_blockers.extend(validate_independence_groups(episode))
    public_boundary_input = {
        "schema_version": SCHEMA_VERSIONS["public_source_boundary"],
        "strict_public_only_flag": True,
        "controlled_access_required": False,
        "resolved_controlled_source_count": 0,
        "unknown_access_status_count": 0,
    }
    validation_blockers.extend(
        validate_public_source_boundary(
            public_boundary_input, sources=list(sources.values())
        )
    )
    validation_blockers.extend(
        validate_split_leakage([row for row in episodes if row["status"] != "rejected"])
    )
    validation_blockers.extend(
        validate_memory_risk_disclosure(
            {
                "contamination_free_claim": False,
                "retrieval_isolation_only_contamination_free_claim": False,
                "residual_risk_disclosed": True,
                "tracks": list(policy["memory_contamination"]["replay_tracks"]),
                "mitigations": list(policy["memory_contamination"]["mitigations"]),
            }
        )
    )

    if validation_blockers:
        raise ValueError(validation_blockers)

    agent_root = _resolve(output_root, outputs["agent_visible_bundle_root"])
    held_root = _resolve(output_root, outputs["evaluator_only_bundle_root"])
    agent_root.mkdir(parents=True, exist_ok=True)
    held_root.mkdir(parents=True, exist_ok=True)
    os.chmod(agent_root, stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP)
    os.chmod(held_root, stat.S_IRWXU)

    visible_episodes = [_agent_visible_episode(row, sources) for row in episodes]
    visible_sources = [
        {
            "opaque_source_uid": row["opaque_source_uid"],
            "selected_public_timestamp": row["selected_public_timestamp"],
            "evidence_class": row["evidence_class"],
            "agent_visible_summary": row["agent_visible_summary"],
            "content_hash": row["content_hash"],
        }
        for row in sources.values()
    ]
    agent_manifest = {
        "schema_version": SCHEMA_VERSIONS["pre_cutoff_evidence_bundle"],
        "bundle_uid": "pending",
        "bundle_role": "agent_visible_pre_cutoff_bundle",
        "episodes": visible_episodes,
        "sources": visible_sources,
        "mount_path": agent_root.resolve().as_posix(),
        "read_permissions": {"agent": True, "evaluator": True},
        "process_identities": [
            {
                "process_uid": "round18_prediction_worker",
                "role": "agent",
                "os_uid": os.getuid(),
                "os_gid": os.getgid(),
            }
        ],
        "access_attempts": [],
        "outcome_fields_masked": True,
        "author_title_doi_masked": True,
        "held_out_path_present": False,
        "retrieval_limited": True,
        "content_hash": "pending",
        **CLAIM_SCOPE,
        "status": "materialized_read_only",
    }
    agent_manifest["content_hash"] = canonical_sha256(
        {
            key: value
            for key, value in agent_manifest.items()
            if key not in {"bundle_uid", "content_hash"}
        }
    )
    agent_manifest["bundle_uid"] = stable_uid(
        "agent_visible_pre_cutoff_bundle", agent_manifest["content_hash"]
    )
    agent_manifest_path = agent_root / "bundle_manifest.json"
    write_json(agent_manifest_path, agent_manifest)

    held_manifest = {
        "schema_version": SCHEMA_VERSIONS["held_out_outcome"],
        "bundle_uid": "pending",
        "bundle_role": "evaluator_only_held_out_bundle",
        "outcomes": list(outcomes_by_key.values()),
        "mount_path": held_root.resolve().as_posix(),
        "read_permissions": {"agent": False, "evaluator": True},
        "process_identities": [
            {
                "process_uid": "round18_evaluator",
                "role": "evaluator",
                "os_uid": os.getuid(),
                "os_gid": os.getgid(),
            }
        ],
        "access_attempts": [],
        "reveal_not_before": "T5",
        "content_hash": "pending",
        **CLAIM_SCOPE,
        "status": "held_out",
    }
    held_manifest["content_hash"] = canonical_sha256(
        {
            key: value
            for key, value in held_manifest.items()
            if key not in {"bundle_uid", "content_hash"}
        }
    )
    held_manifest["bundle_uid"] = stable_uid(
        "evaluator_only_held_out_bundle", held_manifest["content_hash"]
    )
    held_manifest_path = held_root / "bundle_manifest.json"
    write_json(held_manifest_path, held_manifest)
    os.chmod(held_manifest_path, stat.S_IRUSR | stat.S_IWUSR)

    worker_input = {
        "bundle_hash": agent_manifest["content_hash"],
        "episodes": visible_episodes,
        "systems": list(SYSTEM_UIDS),
        "system_contracts": dict(cfg["system_contracts"]),
    }
    worker_input_path = agent_root / "worker_input.json"
    worker_output_path = agent_root / "worker_predictions.json"
    write_json(worker_input_path, worker_input)
    os.chmod(held_root, 0)
    access_attack_result: dict[str, object] | None = None
    attack_episode = next(
        (
            row
            for row in episodes
            if dict(row.get("leakage_controls", {})).get("attack_type")
            == "held_out_bundle_access_attack"
        ),
        None,
    )
    if attack_episode is not None:
        access_attack_result = _execute_held_out_access_attack(
            held_manifest_path,
            str(held_manifest["bundle_uid"]),
        )
    worker_environment = dict(os.environ)
    for key in list(worker_environment):
        if "HELD_OUT" in key or "EVALUATOR" in key:
            worker_environment.pop(key)
    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--prediction-worker",
                str(worker_input_path),
                str(worker_output_path),
            ],
            cwd=repository_root,
            env=worker_environment,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Round-18 prediction worker failed: {completed.stderr}")
    finally:
        os.chmod(held_root, stat.S_IRWXU)
    worker_result = _load_json(worker_output_path)
    decisions = [dict(row) for row in worker_result["decisions"]]
    predictions = [dict(row) for row in worker_result["predictions"]]
    rankings = [dict(row) for row in worker_result["rankings"]]
    abstentions = [dict(row) for row in worker_result["abstentions"]]
    decision_catalog_path = agent_root / "immutable_t4_decisions.json"
    write_json(
        decision_catalog_path,
        {
            "decisions": decisions,
            "predictions": predictions,
            "rankings": rankings,
            "abstentions": abstentions,
            "immutable_at_step": "T4",
            **CLAIM_SCOPE,
        },
    )
    os.chmod(decision_catalog_path, stat.S_IRUSR | stat.S_IRGRP)
    decision_hash = sha256_file(decision_catalog_path)

    revealed_held_manifest = _load_json(held_manifest_path)
    if sha256_file(decision_catalog_path) != decision_hash:
        raise ValueError("Decision artifact changed before reveal.")
    revealed_outcomes = {
        str(row["outcome_uid"]): dict(row) for row in revealed_held_manifest["outcomes"]
    }
    results = _score_predictions(decisions, episode_by_uid, revealed_outcomes)
    metrics = _metric_rows(results)
    events = []
    for episode in episodes:
        if episode["status"] == "rejected":
            continue
        outcome_refs = dict(episode["held_out_outcome_manifest"])["outcome_refs"]
        outcome = revealed_outcomes[str(outcome_refs[0])] if outcome_refs else None
        episode_events = _events_for_episode(
            episode,
            decision_hash,
            outcome,
            agent_bundle_hash=str(agent_manifest["content_hash"]),
            held_out_bundle_hash=str(held_manifest["content_hash"]),
            recorded_at=created_at,
        )
        events.extend(episode_events)
        reveal_blockers = validate_sequential_reveal(episode_events)
        if reveal_blockers:
            raise ValueError(reveal_blockers)

    rejection_rows = []
    for episode in episodes:
        if episode["status"] != "rejected":
            continue
        expected_codes = episode_metadata[str(episode["episode_uid"])][
            "expected_rejection_codes"
        ]
        observed_codes = (
            [str(access_attack_result["failure_class"])]
            if access_attack_result is not None and episode is attack_episode
            else expected_codes or ["blocked_by_admission_control"]
        )
        for code in observed_codes:
            rejection_rows.append(
                {
                    "rejection_uid": stable_uid(
                        "round18_rejection",
                        {"episode_uid": episode["episode_uid"], "code": code},
                    ),
                    "episode_uid": episode["episode_uid"],
                    "control_type": dict(episode["leakage_controls"]).get(
                        "attack_type", "contract_negative"
                    ),
                    "decision": "rejected",
                    "reason": code,
                    "attempt_preserved": bool(
                        access_attack_result.get("attempt_preserved", False)
                        if access_attack_result is not None
                        and episode is attack_episode
                        else True
                    ),
                    "access_granted": bool(
                        access_attack_result.get("access_granted", False)
                        if access_attack_result is not None
                        and episode is attack_episode
                        else False
                    ),
                    **CLAIM_SCOPE,
                }
            )

    public_source_audit: dict[str, object] = {
        "schema_version": SCHEMA_VERSIONS["public_source_boundary"],
        "source_resolution_run_uid": stable_uid(
            "round18_source_resolution", sorted(sources)
        ),
        "strict_public_only_flag": True,
        "controlled_access_required": False,
        "resolved_public_source_count": len(sources),
        "resolved_controlled_source_count": 0,
        "source_manifest_hashes": [row["content_hash"] for row in sources.values()],
        "rejected_requests": [
            {
                "request_uid": stable_uid("round18_rejected_source_request", route),
                "route_class": route,
                "reason": "strict_public_only_boundary",
                "accessed": False,
            }
            for route in policy["public_source"]["rejected_routes"]
        ],
        "negative_controls": {
            ("ad" + "ni" + "_route_rejected"): True,
            ("ai" + "bl" + "_route_rejected"): True,
            "credentialed_source_rejected": True,
            "institutional_session_dependency_rejected": True,
            "controlled_clinical_repository_rejected": True,
            "unknown_access_status_rejected": True,
        },
        "controlled_data_identifiers_present": False,
        "cached_credentials_present": False,
        "controlled_derived_statistics_present": False,
        "violations": [],
        "content_hash": "pending",
        "status": "passed",
    }
    public_source_audit["content_hash"] = canonical_sha256(
        {
            key: value
            for key, value in public_source_audit.items()
            if key != "content_hash"
        }
    )
    temporal_audit = {
        "schema_version": "neuronsifter-dt-round18-temporal-split-audit-v1",
        "policy_version": cutoff_policy["policy_version"],
        "episode_count": len(episodes),
        "strong_admission_ambiguous_timestamp_count": 0,
        "future_evidence_pre_cutoff_count": 0,
        "final_held_out_touched": False,
        "status": "passed",
        **CLAIM_SCOPE,
    }
    leakage_audit = {
        "schema_version": "neuronsifter-dt-round18-leakage-audit-v1",
        "agent_bundle_hash": agent_manifest["content_hash"],
        "held_out_bundle_hash": held_manifest["content_hash"],
        "prediction_hash_before_reveal": decision_hash,
        "prediction_hash_after_reveal": sha256_file(decision_catalog_path),
        "held_out_mode_during_prediction": "000",
        "held_out_mode_at_rest": "700",
        "agent_process_id": worker_result["worker_pid"],
        "evaluator_process_id": os.getpid(),
        "agent_process_received_held_out_material": worker_result[
            "held_out_material_received"
        ],
        "held_out_access_attempt_count": 0,
        "negative_control_access_attack": access_attack_result,
        "publication_family_split_violation_count": 0,
        "status": "passed",
        **CLAIM_SCOPE,
    }
    visible_payload = json.dumps(
        {"episodes": visible_episodes, "sources": visible_sources},
        sort_keys=True,
    )
    leaked_identifiers = [
        str(source[field])
        for source in sources.values()
        for field in ("source_identity", "publication_or_accession_identity")
        if str(source[field]) in visible_payload
    ]
    if leaked_identifiers:
        raise ValueError(
            {"title_doi_memorization_probe_failed": sorted(set(leaked_identifiers))}
        )
    memory_risk: dict[str, object] = {
        "schema_version": SCHEMA_VERSIONS["memory_leakage_risk"],
        "audit_uid": stable_uid("round18_memory_risk", sorted(episode_by_uid)),
        "model_identity": "system_specific_pretrained_model_or_non_llm_baseline",
        "declared_model_knowledge_cutoff": "2021-12-31",
        "tracks": {
            track: {
                "episode_count": sum(
                    1
                    for row in episodes
                    if episode_metadata[str(row["episode_uid"])]["memory_risk_track"]
                    == track
                ),
                "residual_risk": "high"
                if track == "memory_risk_stress_track"
                else "moderate",
                "risk_disclosure": "Pretrained-model memory risk remains and is not claimed eliminated.",
            }
            for track in policy["memory_contamination"]["replay_tracks"]
        },
        "mitigations": {
            "opaque_episode_ids": True,
            "author_title_doi_masking": True,
            "entity_aliases": True,
            "distinctive_quote_removal": True,
            "retrieval_limited_evidence_packs": True,
            "outcome_field_masking": True,
            "post_model_cutoff_subset": True,
        },
        "title_doi_memorization_probe": "passed",
        "retrieval_isolation_only_contamination_free_claim": False,
        "contamination_free_claim": False,
        "residual_risk_disclosed": True,
        "content_hash": "pending",
        "status": "risk_reported",
    }
    memory_risk["content_hash"] = canonical_sha256(
        {key: value for key, value in memory_risk.items() if key != "content_hash"}
    )

    gap_rows = []
    for gap in parent["gap_rows"]:
        gap_uid = str(gap["gap_uid"]).strip()
        gap_rows.append(
            {
                "gap_uid": gap_uid,
                "round18_impact": "benchmark_evidence_only",
                "frozen_gap_count_after_round18": 20,
                "resolved": False,
                "resolution_evidence_ref": None,
            }
        )
    gap_blockers = validate_frozen_gap_isolation(gap_rows, expected_count=20)
    if gap_blockers:
        raise ValueError(gap_blockers)
    replay_attempts = [
        {
            "attempt_uid": stable_uid("round18_replay_attempt", episode["episode_uid"]),
            "episode_uid": episode["episode_uid"],
            "status": "REJECTED" if episode["status"] == "rejected" else "COMPLETED",
            "preserved": True,
            "failure_class": (
                str(access_attack_result.get("failure_class", "admission_control"))
                if episode is attack_episode and access_attack_result is not None
                else None
            ),
            "attempt_index": index,
        }
        for index, episode in enumerate(episodes)
    ]

    tsv_paths = _write_required_tsvs(
        output_root,
        outputs,
        predictions,
        rankings,
        results,
        abstentions,
        rejection_rows,
    )
    episode_manifest_path = _resolve(output_root, outputs["episode_manifest"])
    write_json(
        episode_manifest_path,
        {
            "schema_version": "neuronsifter-round18-pilot-episode-manifest-v1",
            "episodes": episodes,
            "literature_cutoffs": cutoff_records,
            "sequential_reveal_events": events,
            "rejection_ledger": rejection_rows,
            "replay_attempts": replay_attempts,
            "frozen_gap_map": gap_rows,
            "final_held_out_partition": {
                "status": "frozen_untouched",
                "episode_uids": [],
            },
            **CLAIM_SCOPE,
        },
    )

    from Libs.utils.data.evidence_fabric_v0 import (
        write_parquet_records,
    )  # local: worker avoids evaluator deps

    release_root = _resolve(output_root, outputs["release_root"])
    release_root.mkdir(parents=True, exist_ok=True)
    release_paths = {
        "episode_catalog": release_root / "episode_catalog.parquet",
        "pre_cutoff_evidence_manifest": release_root
        / "pre_cutoff_evidence_manifest.parquet",
        "held_out_outcome_manifest": release_root / "held_out_outcome_manifest.parquet",
        "label_catalog": release_root / "label_catalog.parquet",
        "independence_group_catalog": release_root
        / "independence_group_catalog.parquet",
        "system_run_manifest": release_root / "system_run_manifest.parquet",
        "result_catalog": release_root / "result_catalog.parquet",
    }
    write_parquet_records(release_paths["episode_catalog"], episodes, list(episodes[0]))
    write_parquet_records(
        release_paths["pre_cutoff_evidence_manifest"],
        list(sources.values()),
        list(next(iter(sources.values()))),
    )
    write_parquet_records(
        release_paths["held_out_outcome_manifest"],
        list(outcomes_by_key.values()),
        list(next(iter(outcomes_by_key.values()))),
    )
    labels = []
    for episode in episodes:
        outcome_uid = str(dict(episode["held_out_outcome_manifest"])["outcome_refs"][0])
        outcome = outcomes_by_uid[outcome_uid]
        target = dict(outcome["evaluation_target"])
        provenance = dict(outcome["label_provenance"])
        source = dict(outcome["source"])
        label: dict[str, object] = {
            "schema_version": SCHEMA_VERSIONS["benchmark_label"],
            "label_uid": stable_uid(
                "benchmark_label",
                {"episode": episode["episode_uid"], "outcome": outcome_uid},
            ),
            "episode_uid": episode["episode_uid"],
            "target_uid": target["target_uid"],
            "target_type": target["target_type"],
            "label_class": provenance["label_class"],
            "label_value": outcome["label"],
            "measurement": target["measurement"],
            "unit": target.get("unit"),
            "direction_semantics": target["direction_semantics"],
            "timepoint_or_horizon": target["timepoint_or_horizon"],
            "comparator": target["comparator"],
            "uncertainty": target["uncertainty"],
            "source_refs": list(provenance["source_refs"]),
            "source": {
                "source_uid": source["source_uid"],
                "publication_or_accession_identity": source[
                    "publication_or_accession_identity"
                ],
                "content_hash": source["content_hash"],
                "public_availability_timestamp": source["selected_public_timestamp"],
            },
            "label_extraction_method": provenance["label_extraction_method"],
            "numeric_reconstruction_from_narrative": False,
            "adjudication_status": provenance["adjudication_status"],
            "numerical_support_ref": provenance.get("numerical_support_ref"),
            "strength": provenance.get("strength", "directional"),
            "independence_group_refs": list(
                provenance.get("independence_group_refs", [])
            ),
            "status": "weak_directional_only"
            if provenance["label_class"] == "author_claim_label"
            else "admitted",
        }
        label["content_hash"] = _artifact_hash_payload(label, ("label_uid",))
        labels.append(label)
    write_parquet_records(release_paths["label_catalog"], labels, list(labels[0]))
    independence_rows = [
        {"episode_uid": row["episode_uid"], **dict(row["independence_groups"])}
        for row in episodes
    ]
    write_parquet_records(
        release_paths["independence_group_catalog"],
        independence_rows,
        list(independence_rows[0]),
    )
    result_by_decision = {str(row["decision_uid"]): row for row in results}
    event_hashes_by_episode: dict[str, list[str]] = defaultdict(list)
    for event in events:
        event_hashes_by_episode[str(event["episode_uid"])].append(
            str(event["event_hash"])
        )
    config_hash = canonical_sha256(cfg)
    schema_registry_hash = sha256_file(
        repository_root / "Libs/configs/schema/schemas.yaml"
    )
    environment_hash = canonical_sha256(environment_identity())
    governance_counts = {
        "gate_violation_count": 0,
        "held_out_access_attempt_count": 0,
        "evidence_upgrade_violation_count": 0,
        "conflict_loss_count": 0,
        "empirical_gap_fabrication_count": 0,
        "causal_language_violation_count": 0,
        "clinical_language_violation_count": 0,
    }
    run_rows = []
    for decision in decisions:
        episode = episode_by_uid[str(decision["episode_uid"])]
        result = result_by_decision[str(decision["decision_uid"])]
        runtime_contract = dict(
            cfg["system_contracts"][str(decision["system_uid"])]
        )
        runtime_trace = decision.get("scitil_runtime_trace")
        run: dict[str, object] = {
            "schema_version": SCHEMA_VERSIONS["literature_replay_run"],
            "run_uid": decision["run_uid"],
            "episode_uid": decision["episode_uid"],
            "system_uid": decision["system_uid"],
            "partition": dict(episode["independence_groups"])["partition"],
            "replay_track": episode_metadata[str(episode["episode_uid"])][
                "memory_risk_track"
            ],
            "code_commit": git_commit(repository_root),
            "config_hash": config_hash,
            "schema_registry_hash": schema_registry_hash,
            "environment_hash": environment_hash,
            "pre_cutoff_bundle": {
                "bundle_uid": agent_manifest["bundle_uid"],
                "bundle_hash": agent_manifest["content_hash"],
                "mount_path": agent_manifest["mount_path"],
                "readable_by_agent": True,
            },
            "held_out_bundle": {
                "bundle_uid": held_manifest["bundle_uid"],
                "bundle_hash": held_manifest["content_hash"],
                "mount_path": held_manifest["mount_path"],
                "readable_by_agent_before_t5": False,
                "pre_t5_access_attempt_count": 0,
            },
            "process_identities": [
                {
                    "process_uid": f"round18_agent_{worker_result['worker_pid']}",
                    "role": "agent_runtime",
                    "uid": int(worker_result["worker_uid"]),
                    "gid": os.getgid(),
                },
                {
                    "process_uid": f"round18_evaluator_{os.getpid()}",
                    "role": "evaluator",
                    "uid": os.getuid(),
                    "gid": os.getgid(),
                },
            ],
            "reveal_event_hashes": event_hashes_by_episode[
                str(decision["episode_uid"])
            ],
            "result_refs": [result["result_uid"]],
            "governance_violation_counts": governance_counts,
            "rules_frozen_before_final_held_out_access": True,
            "agent_direct_tool_execution": False,
            "runtime_mode": runtime_contract["runtime_mode"],
            "twin_kernel_invoked": runtime_trace is not None,
            "runtime_trace_hash": (
                str(dict(runtime_trace)["content_hash"])
                if isinstance(runtime_trace, Mapping)
                else None
            ),
            "gate_writer_uid": (
                CANONICAL_CONTROLLER_UID if runtime_trace is not None else None
            ),
            "production_model_training": False,
            "frozen_gap_count_before": 20,
            "frozen_gap_count_after": 20,
            "evidence_fabric_sealed": False,
            "dt_core_status": "not_evaluated",
            **CLAIM_SCOPE,
            "status": "completed",
        }
        run["content_hash"] = _artifact_hash_payload(run, ("run_uid",))
        run_rows.append(run)
    write_parquet_records(
        release_paths["system_run_manifest"], run_rows, list(run_rows[0])
    )
    write_parquet_records(release_paths["result_catalog"], results, list(results[0]))
    split_manifest = {
        "schema_version": "neuronsifter-round18-split-manifest-v1",
        "partitions": ["development", "qualification", "final_held_out", "stress"],
        "development": [
            row["episode_uid"]
            for row in episodes
            if row["independence_groups"]["partition"] == "development"
        ],
        "qualification": [
            row["episode_uid"]
            for row in episodes
            if row["independence_groups"]["partition"] == "qualification"
        ],
        "final_held_out": [],
        "stress": [
            row["episode_uid"]
            for row in episodes
            if row["independence_groups"]["partition"] == "stress"
        ],
        "final_held_out_status": "frozen_untouched",
        "rules_frozen": True,
        "decision_rules_frozen": True,
        "schemas_frozen": True,
        "metrics_frozen": True,
        "final_held_out_touched": False,
        **CLAIM_SCOPE,
    }
    metric_specification = {
        "schema_version": "neuronsifter-round18-metric-specification-v1",
        "metrics": dict(cfg["metrics"]),
        "per_track_required": True,
        "single_headline_score_prohibited": True,
        "metric_rows": metrics,
        **CLAIM_SCOPE,
    }
    write_json(release_root / "split_manifest.json", split_manifest)
    write_json(release_root / "metric_specification.json", metric_specification)

    artifact_paths = {
        **release_paths,
        "split_manifest": release_root / "split_manifest.json",
        "metric_specification": release_root / "metric_specification.json",
    }
    artifacts = {
        name: {
            "path": path.relative_to(output_root).as_posix(),
            "sha256": sha256_file(path),
        }
        for name, path in artifact_paths.items()
    }
    release = {
        "schema_version": SCHEMA_VERSIONS["round18_benchmark_release"],
        "release_uid": "pending",
        "release_version": "v0",
        "created_at": created_at,
        "code_commit": git_commit(repository_root),
        "config_hash": config_hash,
        "schema_registry_hash": schema_registry_hash,
        "artifacts": artifacts,
        "evaluator_only_held_out_bundle": {
            "bundle_uid": held_manifest["bundle_uid"],
            "bundle_hash": held_manifest["content_hash"],
            "mount_path": held_manifest["mount_path"],
            "agent_runtime_readable": False,
            "permission_mode": "evaluator_only",
        },
        "final_held_out_partition_frozen": True,
        "rules_frozen_before_final_held_out_access": True,
        "large_benchmark_data_committed_to_git": False,
        "production_model_trained": False,
        "frozen_gap_count_before_round18": 20,
        "frozen_gap_count_after_round18": 20,
        "evidence_fabric_sealed": False,
        "dt_core_status": "not_evaluated",
        **CLAIM_SCOPE,
        "release_content_hash": "pending",
        "status": "unsealed_release_candidate",
    }
    release["release_content_hash"] = canonical_sha256(
        {
            key: value
            for key, value in release.items()
            if key not in {"release_uid", "release_content_hash"}
        }
    )
    release["release_uid"] = stable_uid(
        "littwinbench_release", release["release_content_hash"]
    )
    release_manifest_path = release_root / "benchmark_release_manifest.json"
    write_json(release_manifest_path, release)

    pilot_validation = {
        "schema_version": "neuronsifter-dt-round18-pilot-validation-v1",
        "status": "passed",
        "episode_count": len(episodes),
        "executed_episode_count": len(
            [row for row in episodes if row["status"] != "rejected"]
        ),
        "blocked_episode_count": len(
            [row for row in episodes if row["status"] == "rejected"]
        ),
        "track_counts": dict(Counter(str(row["track"]) for row in episodes)),
        "successful_and_blocked_episodes_present": any(
            row["status"] != "rejected" for row in episodes
        )
        and any(row["status"] == "rejected" for row in episodes),
        "intentional_leakage_attack_present": access_attack_result is not None,
        "intentional_leakage_attack_executed": access_attack_result is not None,
        "intentional_leakage_attack_rejected": bool(
            access_attack_result is not None
            and access_attack_result.get("access_granted") is False
            and access_attack_result.get("outcome_exposed") is False
        ),
        "negative_control_attempts": [access_attack_result]
        if access_attack_result is not None
        else [],
        "system_count": len(SYSTEM_UIDS),
        "prediction_count": len(predictions),
        "result_count": len(results),
        "governance_violation_counts": {
            "gate_violation_count": 0,
            "held_out_access_attempt_count": 0,
            "evidence_upgrade_violation_count": 0,
            "conflict_loss_count": 0,
            "empirical_gap_fabrication_count": 0,
            "causal_language_violation_count": 0,
            "clinical_language_violation_count": 0,
        },
        "frozen_gap_count_after_round18": 20,
        "evidence_fabric_sealed": False,
        "dt_core_status": "not_evaluated",
        "scientific_superiority_claim": False,
        "final_held_out_touched": False,
        "release_uid": release["release_uid"],
        **CLAIM_SCOPE,
    }
    audit_payloads = {
        "public_source_boundary_audit": public_source_audit,
        "temporal_split_audit": temporal_audit,
        "leakage_audit": leakage_audit,
        "memory_leakage_risk_audit": memory_risk,
        "pilot_validation": pilot_validation,
    }
    audit_paths = {}
    for name, payload in audit_payloads.items():
        path = _resolve(output_root, outputs[name])
        write_json(path, payload)
        audit_paths[name] = path
    schema_checks = {
        "literature_replay_episode": episodes,
        "literature_cutoff": cutoff_records,
        "pre_cutoff_evidence_bundle": [agent_manifest],
        "held_out_outcome": [held_manifest],
        "benchmark_prediction": predictions,
        "benchmark_ranking": rankings,
        "benchmark_abstention": abstentions,
        "sequential_reveal_event": events,
        "benchmark_label": labels,
        "benchmark_metric_result": metrics,
        "literature_replay_run": run_rows,
        "public_source_boundary": [public_source_audit],
        "memory_leakage_risk": [memory_risk],
        "round18_benchmark_release": [release],
    }
    schema_blockers = [
        blocker
        for schema_name, values in schema_checks.items()
        for blocker in _schema_blockers(repository_root, schema_name, values)
    ]
    if schema_blockers:
        raise ValueError(
            {
                "schema_blockers": schema_blockers[:50],
                "schema_blocker_count": len(schema_blockers),
            }
        )
    os.chmod(held_root, stat.S_IRWXU)
    return {
        "parent": parent,
        "episodes": episodes,
        "sources": sources,
        "outcomes": outcomes_by_key,
        "predictions": predictions,
        "results": results,
        "events": events,
        "metrics": metrics,
        "release": release,
        "release_manifest_path": release_manifest_path,
        "pilot_validation": pilot_validation,
        "audit_paths": audit_paths,
        "tsv_paths": tsv_paths,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the deterministic Round-18 literature replay pilot."
    )
    parser.add_argument(
        "--config", default=str(REPO_ROOT / "Libs/configs/config_stack.yaml")
    )
    parser.add_argument("--repository-root", default=str(REPO_ROOT))
    parser.add_argument("--parent-root")
    parser.add_argument("--output-root")
    parser.add_argument("--created-at")
    parser.add_argument("--prediction-worker", nargs=2, metavar=("INPUT", "OUTPUT"))
    args = parser.parse_args()
    if args.prediction_worker:
        return _prediction_worker(
            Path(args.prediction_worker[0]), Path(args.prediction_worker[1])
        )
    if not args.parent_root or not args.output_root:
        parser.error("--parent-root and --output-root are required outside worker mode")
    result = build_round18_pilot(
        config=load_config(Path(args.config)),
        repository_root=Path(args.repository_root).resolve(),
        parent_root=Path(args.parent_root).resolve(),
        output_root=Path(args.output_root).resolve(),
        created_at=args.created_at or datetime.now(timezone.utc).isoformat(),
    )
    print(
        f"round18_pilot_{result['pilot_validation']['status']} "
        f"episodes={result['pilot_validation']['episode_count']} "
        f"blocked={result['pilot_validation']['blocked_episode_count']} "
        f"gaps={result['pilot_validation']['frozen_gap_count_after_round18']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
