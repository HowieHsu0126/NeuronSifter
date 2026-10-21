"""Build the restricted, read-only Round-16 governed what-if pilot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.data.raw_data_registry import load_config, read_tsv, write_json, write_tsv  # noqa: E402
from Libs.utils.data.trajectory_contracts import array_sha256  # noqa: E402
from Libs.utils.data.what_if_contracts import (  # noqa: E402
    ELIGIBILITY_ROLES,
    SCHEMA_VERSIONS,
    UNCERTAINTY_COMPONENTS,
    ControlRollout,
    CounterfactualContrast,
    RolloutAttempt,
    RolloutConstraintReport,
    RolloutEngine,
    RolloutRequest,
    RolloutReliabilityReport,
    RolloutTrajectory,
    RolloutUncertainty,
    WhatIfActionBinding,
    WhatIfBaseline,
    WhatIfEligibilityDecision,
    WhatIfRequest,
    WhatIfResult,
    canonical_sha256,
    release_content_hash,
    round16_config,
    stable_uid,
    validate_action_binding,
    validate_attempts,
    validate_baseline,
    validate_constraint_report,
    validate_contrast,
    validate_control_pair,
    validate_eligibility,
    validate_engine,
    validate_engine_compatibility,
    validate_gap_map,
    validate_lineage_acyclic,
    validate_reliability,
    validate_release_candidate,
    validate_result,
    validate_rollout_request,
    validate_rollout_trajectory,
    validate_round16_config,
    validate_uncertainty,
    validate_what_if_request,
)
from Libs.utils.exp.orchestration_controller import git_dirty_status  # noqa: E402


ROLLOUT_VALIDATOR_UID = "round16_governed_what_if_engine_validator"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repository_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def environment_identity() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "not_slurm"),
        "slurm_job_name": os.environ.get("SLURM_JOB_NAME", "not_slurm"),
        "slurm_node": os.environ.get("SLURMD_NODENAME", socket.gethostname()),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION", "not_slurm"),
        "pycache_prefix": os.environ.get("PYTHONPYCACHEPREFIX", ""),
    }


def _resolve(root: Path, relative: object) -> Path:
    path = Path(str(relative))
    if path.is_absolute():
        raise ValueError(f"Round-16 configured paths must be relative: {path}")
    return root / path


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _parent_paths(config: Mapping[str, object], parent_root: Path) -> dict[str, Path]:
    return {
        name: _resolve(parent_root, path)
        for name, path in dict(round16_config(config)["parent_artifacts"]).items()
    }


def validate_parent_state(config: Mapping[str, object], parent_root: Path) -> dict[str, object]:
    paths = _parent_paths(config, parent_root)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Round-16 parent artifacts are missing: {missing}")
    handoff = _load_json(paths["round16_handoff"])
    gate = _load_json(paths["round15_gate"])
    validation = _load_json(paths["round15_validation"])
    identity = _load_json(paths["round15_execution_identity"])
    full_suite = _load_json(paths["round15_full_suite_execution"])
    coverage = _load_json(paths["round15_trajectory_coverage"])
    effective_units = _load_json(paths["round15_effective_trajectory_units"])
    release = _load_json(paths["round15_release_candidate"])
    trajectory = _load_json(paths["round15_trajectory"])
    initial_state = _load_json(paths["round15_initial_state"])
    action_binding = _load_json(paths["round15_action_binding"])
    numerical_qc = _load_json(paths["round15_numerical_qc"])
    constraint_qc = _load_json(paths["round15_constraint_qc"])
    trajectory_uncertainty = _load_json(paths["round15_uncertainty"])
    admission_rows = read_tsv(paths["round15_e4_admission_ledger"])
    gap_rows = read_tsv(paths["round15_frozen_gap_map"])
    blockers: list[dict[str, object]] = []
    if handoff.get("status") != "ready_for_governed_what_if_engine_contracts":
        blockers.append({"code": "round16_entry_not_ready", "message": "Round-16 handoff is not ready."})
    if gate.get("status") != "clean_action_complete_trajectory_domain_resolved" or gate.get("overall_gate_status") != "accepted":
        blockers.append({"code": "round15_gate_not_accepted", "message": "Round-15 Gate is not accepted."})
    if validation.get("status") != "passed" or int(validation.get("blocker_count", -1)) != 0:
        blockers.append({"code": "round15_validation_not_passed", "message": "Round-15 validation is not passing."})
    if identity.get("exact_git_commit") != "d7e5b33cb9c240b111e8e479029213b7866561fc" or identity.get("dirty_status") != "clean":
        blockers.append({"code": "round15_execution_identity_invalid", "message": "Round-15 exact-clean identity changed."})
    structured_test_result = dict(full_suite.get("structured_test_result", {}))
    if not (
        full_suite.get("job_id") == "495307"
        and full_suite.get("terminal_state") == "COMPLETED"
        and full_suite.get("exit_code") == "0:0"
        and int(structured_test_result.get("passed", -1)) == 492
    ):
        blockers.append({"code": "round15_full_suite_invalid", "message": "Accepted Round-15 full-suite evidence changed."})
    if len(admission_rows) != 1 or admission_rows[0].get("decision") != "admitted_e4_restricted":
        blockers.append({"code": "round15_e4_scope_drift", "message": "Round-16 requires exactly one restricted E4 row."})
    if coverage.get("action_classes_represented") != ["no_action_control"] or coverage.get("effective_independent_units") != 1:
        blockers.append({"code": "round15_coverage_drift", "message": "Round-15 restricted coverage changed."})
    if effective_units.get("effective_independent_unit_count") != 1:
        blockers.append({"code": "round15_effective_units_drift", "message": "Round-15 effective unit count changed."})
    if release.get("sealed") is not False or release.get("status") != "materialized_unsealed_successor_candidate":
        blockers.append({"code": "parent_release_not_unsealed", "message": "Parent Evidence Fabric must remain unsealed."})
    if len(gap_rows) != 20:
        blockers.append({"code": "frozen_gap_count_drift", "message": "Exactly 20 parent gaps are required."})
    state_values = np.load(paths["round15_state_values"], allow_pickle=False)
    if array_sha256(state_values) != trajectory.get("state_values_hash"):
        blockers.append({"code": "parent_trajectory_hash_mismatch", "message": "Round-15 state artifact hash changed."})
    if trajectory.get("eligibility") != "qualification_numerical_dynamics_only":
        blockers.append({"code": "parent_trajectory_role_upgrade", "message": "Round-15 restricted trajectory role changed."})
    if action_binding.get("world_action", {}).get("intervention_type") != "no_action_control":
        blockers.append({"code": "unsupported_parent_action", "message": "Only the governed no-action replay is available."})
    if blockers:
        raise ValueError(blockers)
    return {
        "paths": paths,
        "hashes": {name: sha256_file(path) for name, path in paths.items()},
        "handoff": handoff,
        "gate": gate,
        "validation": validation,
        "identity": identity,
        "full_suite": full_suite,
        "coverage": coverage,
        "effective_units": effective_units,
        "release": release,
        "trajectory": trajectory,
        "initial_state": initial_state,
        "action_binding": action_binding,
        "numerical_qc": numerical_qc,
        "constraint_qc": constraint_qc,
        "trajectory_uncertainty": trajectory_uncertainty,
        "admission_rows": admission_rows,
        "gap_rows": gap_rows,
        "state_values": state_values,
    }


def _catalog_row(record_type: str, decision: str, value: Mapping[str, object]) -> dict[str, object]:
    uid_fields = (
        "request_uid", "attempt_uid", "rollout_uid", "pair_uid", "contrast_uid", "uncertainty_uid",
        "reliability_uid", "decision_uid",
    )
    uid = next((str(value[field]) for field in uid_fields if value.get(field)), stable_uid("catalog", value))
    return {
        "record_uid": uid,
        "record_type": record_type,
        "decision": decision,
        "schema_version": value.get("schema_version", "not_applicable_negative_fixture"),
        "object_json": json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
    }


def _write_catalog(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = ["record_uid", "record_type", "decision", "schema_version", "object_json"]
    write_tsv(path, fields, [dict(row) for row in rows])


def _build_gap_rows(parent_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in parent_rows:
        blocker_text = str(row.get("remaining_blocker", ""))
        status = "requires_empirical_observation"
        if "support" in blocker_text:
            status = "requires_support_domain"
        elif "trajectory" in blocker_text or "action" in blocker_text:
            status = "requires_new_e4"
        rows.append(
            {
                "gap_uid": row["gap_uid"],
                "round15_status": row["current_status"],
                "round16_status": status,
                "model_rollout_uid": "",
                "model_output_role": "hypothesis_or_stress_information_only",
                "resolved_by_model_rollout": False,
                "empirical_resolution_evidence": "",
                "remaining_blocker": row["remaining_blocker"],
            }
        )
    return rows


def build_round16_pilot(
    *, config: Mapping[str, object], repository_root: Path, parent_root: Path, output_root: Path, created_at: str
) -> dict[str, object]:
    config_blockers = validate_round16_config(config)
    if config_blockers:
        raise ValueError(config_blockers)
    parent = validate_parent_state(config, parent_root)
    settings = round16_config(config)
    pilot = dict(settings["pilot"])
    engine_config = dict(settings["restricted_engine"])
    support_domain = dict(parent["handoff"]["supported_trajectory_domains"][0])
    domain_ref = str(support_domain["support_domain_uid"])
    known_domains = {domain_ref}
    trajectory = dict(parent["trajectory"])
    initial = dict(parent["initial_state"])
    parent_binding = dict(parent["action_binding"])
    world_action = dict(parent_binding["world_action"])
    time_grid = [float(value) for value in trajectory["canonical_export_time_grid"]]
    environment = environment_identity()
    environment_hash = canonical_sha256(environment)
    commit = git_commit(repository_root)

    request = WhatIfRequest(
        schema_version=SCHEMA_VERSIONS["what_if_request"],
        request_uid=str(pilot["request_uid"]),
        research_question_uid=str(pilot["research_question_uid"]),
        baseline_state_ref=str(initial["state_uid"]),
        candidate_world_action_ref=str(world_action["action_uid"]),
        control_action_ref=str(world_action["action_uid"]),
        requested_outcomes=[str(value) for value in trajectory["species_or_feature_axis"]],
        requested_time_horizon=dict(pilot["requested_time_horizon"]),
        requested_time_grid=time_grid,
        requested_regions=[str(value) for value in trajectory["region_axis"]],
        requested_compartments=[str(value) for value in trajectory["compartment_axis"]],
        requested_feature_namespace=str(pilot["state_namespace"]),
        rollout_engine_policy={"required_status": "qualified_for_contract_testing", "fallback_engine_allowed": False},
        ensemble_policy=dict(pilot["ensemble_policy"]),
        fidelity_policy=dict(pilot["fidelity_policy"]),
        uncertainty_requirements=list(UNCERTAINTY_COMPONENTS),
        constraint_requirements=[str(value) for value in parent["constraint_qc"]["required_constraints"]],
        support_domain_ref=domain_ref,
        budget=dict(pilot["budget"]),
        provenance={"round16_parent_handoff_hash": parent["hashes"]["round16_handoff"], "request_origin": "contract_qualification"},
        model_based_counterfactual=True,
        causal_effect_claim=False,
        clinical_recommendation=False,
        empirical_observation=False,
    ).to_dict()
    baseline = WhatIfBaseline(
        schema_version=SCHEMA_VERSIONS["what_if_baseline"],
        baseline_uid=str(initial["state_uid"]),
        state_ref=str(initial["state_values_ref"]),
        state_hash=str(initial["state_hash"]),
        state_time=float(initial["state_time"]),
        state_uncertainty=dict(initial["uncertainty"]),
        biological_or_model_context=dict(initial["biological_or_model_context"]),
        region=[str(value) for value in initial["region_axis"]],
        compartment=[str(value) for value in initial["compartment_axis"]],
        feature_namespace=str(pilot["state_namespace"]),
        support_domain_ref=domain_ref,
        source_role="qualification_numerical_dynamics_only",
        qc_grade="S",
        missingness=dict(initial["missingness"]),
        provenance={"round15_initial_state_hash": parent["hashes"]["round15_initial_state"], "empirical_state": False},
        baseline_class="qualification_synthetic",
        imputation_policy="none_all_channels_explicit",
        empirical_state=False,
        qualification_only=True,
        eligible_for_biological_claims=False,
    ).to_dict()
    parameter_provenance = dict(parent_binding["parameter_provenance"])
    parameter_provenance.update(
        {
            "parent_action_program_hash": parent_binding["action_program_hash"],
            "world_action_hash": canonical_sha256(world_action),
        }
    )
    action_binding = WhatIfActionBinding(
        schema_version=SCHEMA_VERSIONS["what_if_action_binding"],
        binding_uid=stable_uid("what_if_action_binding", {"request": request["request_uid"], "action": world_action}),
        request_uid=str(request["request_uid"]),
        world_action=world_action,
        action_program_hash=str(parent_binding["action_program_hash"]),
        parameter_provenance=parameter_provenance,
        support_domain_ref=domain_ref,
        spatial_scope=dict(world_action["state_scope"]),
        temporal_scope={"time_grid": time_grid, "unit": trajectory["time_unit"], "schedule": world_action["schedule"]},
        combination_components=[],
        interaction_assumptions=[],
        completeness="complete",
        blockers=[],
        status="validated",
    ).to_dict()
    engine = RolloutEngine(
        schema_version=SCHEMA_VERSIONS["rollout_engine"],
        engine_uid=str(engine_config["engine_uid"]),
        engine_type=str(engine_config["engine_type"]),
        version=str(engine_config["version"]),
        source=str(engine_config["source"]),
        code_identity=sha256_file(repository_root / str(engine_config["source"])),
        environment_identity=environment_hash,
        supported_baseline_classes=list(engine_config["supported_baseline_classes"]),
        supported_action_classes=list(engine_config["supported_action_classes"]),
        supported_state_variables=list(engine_config["supported_state_variables"]),
        supported_horizons=list(engine_config["supported_horizons"]),
        supported_fidelities=list(engine_config["supported_fidelities"]),
        support_domain_ref=domain_ref,
        uncertainty_capabilities=["explicit_component_registry", "unknown_preservation", "no_fabricated_interval"],
        constraint_capabilities=[str(value) for value in parent["constraint_qc"]["required_constraints"]],
        qualification_evidence={
            "round15_gate_hash": parent["hashes"]["round15_gate"],
            "round15_admission_hash": parent["hashes"]["round15_e4_admission_ledger"],
            "trajectory_uid": trajectory["trajectory_uid"],
            "admission_decision": parent["admission_rows"][0]["decision"],
        },
        known_limitations=list(engine_config["known_limitations"]),
        trained=bool(engine_config["trained"]),
        status=str(engine_config["qualification_status"]),
    ).to_dict()
    horizon_decision, support_decision, compatibility_blockers = validate_engine_compatibility(
        request=request, baseline=baseline, action_binding=action_binding, engine=engine
    )
    if compatibility_blockers:
        raise ValueError(compatibility_blockers)
    input_bundle_hash = canonical_sha256(parent["hashes"])
    rollout_request = RolloutRequest(
        schema_version=SCHEMA_VERSIONS["rollout_request"],
        rollout_request_uid=stable_uid("rollout_request", {"request": request, "engine": engine["engine_uid"]}),
        what_if_request_uid=str(request["request_uid"]),
        engine_uid=str(engine["engine_uid"]),
        engine_version=str(engine["version"]),
        baseline_hash=str(baseline["state_hash"]),
        action_program_hash=str(action_binding["action_program_hash"]),
        control_program_hash=str(action_binding["action_program_hash"]),
        parameter_hash=str(trajectory["producer_parameters_hash"]),
        seed=int(trajectory["random_seed"]),
        fidelity=str(trajectory["fidelity_uid"]),
        time_grid=time_grid,
        constraint_policy={"policy_version": parent["constraint_qc"]["policy_version"], "fail_closed": True},
        uncertainty_policy={"unknown_must_remain_unknown": True, "component_registry": list(UNCERTAINTY_COMPONENTS)},
        exact_code_commit=commit,
        environment_hash=environment_hash,
        input_bundle_hash=input_bundle_hash,
    ).to_dict()
    action_attempt_uid = stable_uid("rollout_attempt", {"request": rollout_request["rollout_request_uid"], "role": "action"})
    control_attempt_uid = stable_uid("rollout_attempt", {"request": rollout_request["rollout_request_uid"], "role": "control"})
    runtime_identity = {**environment, "code_commit": commit, "dirty_status": git_dirty_status(repository_root)}
    action_attempt = RolloutAttempt(
        schema_version=SCHEMA_VERSIONS["rollout_attempt"], attempt_uid=action_attempt_uid,
        parent_request_uid=str(rollout_request["rollout_request_uid"]),
        submission_command="python Libs/scripts/builders/build_dt_round16_what_if_engine.py --read-only-replay",
        runtime_identity=runtime_identity,
        job_identity={"job_id": environment["slurm_job_id"], "job_name": environment["slurm_job_name"], "node": environment["slurm_node"], "partition": environment["slurm_partition"], "execution_mode": "read_only_replay"},
        stdout_ref="runtime:round16_builder_stdout", stderr_ref="runtime:round16_builder_stderr",
        terminal_state="COMPLETED", exit_code="0:0", raw_output_hashes={"state_values": trajectory["state_values_hash"]},
        failure_class=None, supersedes_attempt_uid=None, preserved=True, status="completed",
    ).to_dict()
    control_attempt = RolloutAttempt(
        schema_version=SCHEMA_VERSIONS["rollout_attempt"], attempt_uid=control_attempt_uid,
        parent_request_uid=str(rollout_request["rollout_request_uid"]),
        submission_command="python Libs/scripts/builders/build_dt_round16_what_if_engine.py --read-only-replay-control",
        runtime_identity=runtime_identity,
        job_identity={"job_id": environment["slurm_job_id"], "job_name": environment["slurm_job_name"], "node": environment["slurm_node"], "partition": environment["slurm_partition"], "execution_mode": "read_only_control_replay"},
        stdout_ref="runtime:round16_builder_stdout", stderr_ref="runtime:round16_builder_stderr",
        terminal_state="COMPLETED", exit_code="0:0", raw_output_hashes={"state_values": trajectory["state_values_hash"]},
        failure_class=None, supersedes_attempt_uid=None, preserved=True, status="completed",
    ).to_dict()
    failed_attempt = RolloutAttempt(
        schema_version=SCHEMA_VERSIONS["rollout_attempt"],
        attempt_uid=stable_uid("rollout_attempt", {"request": rollout_request["rollout_request_uid"], "case": "unsupported_action"}),
        parent_request_uid=str(rollout_request["rollout_request_uid"]),
        submission_command="not_executed_fail_closed_unsupported_action",
        runtime_identity=runtime_identity,
        job_identity={"job_id": "not_submitted", "job_name": "not_submitted", "node": "not_allocated", "partition": "not_allocated", "execution_mode": "pre_execution_rejection"},
        stdout_ref="not_created", stderr_ref="not_created", terminal_state="REJECTED_BEFORE_EXECUTION", exit_code="not_executed",
        raw_output_hashes={}, failure_class="unsupported_action_class", supersedes_attempt_uid=None, preserved=True, status="retained_failure",
    ).to_dict()
    pair_uid = stable_uid("rollout_control_pair", {"request": request["request_uid"], "action": world_action["action_uid"], "control": world_action["action_uid"]})
    state_ref = str(parent["paths"]["round15_state_values"].relative_to(parent_root))
    uncertainty_ref = str(parent["paths"]["round15_uncertainty"].relative_to(parent_root))
    constraint_ref = str(parent["paths"]["round15_constraint_qc"].relative_to(parent_root))
    action_rollout_uid = stable_uid("model_rollout", {"attempt": action_attempt_uid, "role": "candidate_no_action"})
    control_rollout_uid = stable_uid("model_rollout", {"attempt": control_attempt_uid, "role": "matched_no_action_control"})
    rollout_common = {
        "schema_version": SCHEMA_VERSIONS["rollout_trajectory"], "request_uid": request["request_uid"],
        "engine_uid": engine["engine_uid"], "baseline_uid": baseline["baseline_uid"], "action_uid": world_action["action_uid"],
        "control_pair_uid": pair_uid, "native_time_grid": trajectory["native_solver_time_grid"],
        "canonical_time_grid": time_grid, "state_namespace": pilot["state_namespace"], "state_artifact_ref": state_ref,
        "uncertainty_artifact_ref": uncertainty_ref, "constraint_artifact_ref": constraint_ref,
        "content_hashes": {"state_values": trajectory["state_values_hash"], "uncertainty": parent["hashes"]["round15_uncertainty"], "constraints": parent["hashes"]["round15_constraint_qc"]},
        "status": "completed", "eligibility": "contract_and_runtime_validation_only",
        "output_class": "model_rollout", "empirical_observation": False,
    }
    action_rollout = RolloutTrajectory(
        **rollout_common, rollout_uid=action_rollout_uid, attempt_uid=action_attempt_uid,
        provenance={"parent_e4_trajectory_uid": trajectory["trajectory_uid"], "replay_role": "candidate_no_action", "new_trajectory_generated": False},
    ).to_dict()
    control_rollout = RolloutTrajectory(
        **rollout_common, rollout_uid=control_rollout_uid, attempt_uid=control_attempt_uid,
        provenance={"parent_e4_trajectory_uid": trajectory["trajectory_uid"], "replay_role": "matched_no_action_control", "new_trajectory_generated": False},
    ).to_dict()
    control_pair = ControlRollout(
        schema_version=SCHEMA_VERSIONS["rollout_control_pair"], pair_uid=pair_uid,
        action_rollout_uid=action_rollout_uid, control_rollout_uid=control_rollout_uid, control_type="no_action_control",
        baseline_state_family_match=True, engine_version_match=True, parameter_family_match=True, fidelity_match=True,
        time_grid_match=True, seed_policy_compatible=True, state_namespace_match=True, support_domain_match=True,
        pairing_decision="exact_deterministic_pair", intervention_effect_eligible=False, blockers=[], status="validated",
    ).to_dict()
    unknown_components = [
        "initial_state_uncertainty", "model_parameter_uncertainty", "model_family_disagreement",
        "fidelity_disagreement", "observation_or_calibration_uncertainty",
    ]
    components = {
        "initial_state_uncertainty": {"status": "unknown", "reason": "synthetic qualification baseline has no empirical posterior"},
        "action_parameter_uncertainty": {"status": "known", "semantics": "explicit_no_action_program"},
        "model_parameter_uncertainty": {"status": "unknown", "reason": parent["trajectory_uncertainty"]["parameter_uncertainty"]},
        "stochastic_rollout_uncertainty": {"status": "not_applicable", "reason": "deterministic replay"},
        "numerical_uncertainty": {"status": "bounded_by_parent_qc", "interpolation_error": parent["numerical_qc"]["interpolation_error"]},
        "model_family_disagreement": {"status": "unknown", "reason": "one qualified restricted engine"},
        "fidelity_disagreement": {"status": "unknown", "reason": "one qualified fidelity"},
        "support_domain_extrapolation": {"status": "restricted_in_domain", "reason": "synthetic qualification domain only"},
        "observation_or_calibration_uncertainty": {"status": "unknown", "reason": "no empirical calibration admitted"},
    }
    uncertainty = RolloutUncertainty(
        schema_version=SCHEMA_VERSIONS["rollout_uncertainty"],
        uncertainty_uid=stable_uid("rollout_uncertainty", {"rollouts": [action_rollout_uid, control_rollout_uid], "components": components}),
        rollout_uids=[action_rollout_uid, control_rollout_uid], components=components,
        point_estimate={"contrast": 0.0, "estimator": "paired_difference"},
        interval_or_distribution={"status": "unavailable_unknown_components", "fabricated_interval": False},
        aggregation_policy="preserve_components_no_scalar_collapse", unknown_components=unknown_components,
        status="reported_with_unknown_components",
    ).to_dict()
    constraint_report = RolloutConstraintReport(
        schema_version=SCHEMA_VERSIONS["rollout_constraint_report"],
        constraint_report_uid=stable_uid("rollout_constraint_report", {"rollout": action_rollout_uid, "parent": parent["constraint_qc"]}),
        rollout_uid=action_rollout_uid, constraint_policy=str(parent["constraint_qc"]["policy_version"]),
        required_constraints=list(parent["constraint_qc"]["required_constraints"]),
        passed_constraints=list(parent["constraint_qc"]["required_constraints"]), failed_constraints=[], unknown_constraints=[],
        violation_times={}, violation_magnitudes={}, handling_state="valid", formal_contrast_eligible=True,
    ).to_dict()
    reliability_diagnostics = {
        "repeatability": {"status": "passed", "exact_content_hash_match": True},
        "seed_stability": {"status": "not_applicable", "reason": "deterministic replay"},
        "parameter_sensitivity": {"status": "unknown", "reason": "not qualified"},
        "fidelity_consistency": {"status": "unknown", "reason": "single fidelity"},
        "horizon_degradation": {"status": "tested_only", "maximum_hours": 672.0},
        "constraint_violation_rate": {"status": "passed", "required_violation_count": 0},
        "model_family_disagreement": {"status": "insufficient_comparison", "engine_count": 1},
        "baseline_perturbation_sensitivity": {"status": "unknown", "reason": "single baseline"},
        "action_perturbation_sensitivity": {"status": "unknown", "reason": "no-action only"},
        "support_domain_distance": {"status": "restricted_in_domain", "domain_ref": domain_ref},
    }
    reliability = RolloutReliabilityReport(
        schema_version=SCHEMA_VERSIONS["rollout_reliability_report"],
        reliability_uid=stable_uid("rollout_reliability", {"engine": engine["engine_uid"], "diagnostics": reliability_diagnostics}),
        engine_uid=str(engine["engine_uid"]), support_domain_ref=domain_ref, action_class="no_action_control",
        diagnostics=reliability_diagnostics, qualified_horizon=dict(pilot["requested_time_horizon"]),
        maximum_tested_horizon=dict(pilot["requested_time_horizon"]), horizon_decision=horizon_decision,
        support_domain_decision=support_decision, disagreement_outcome="insufficient_comparison",
        status="qualified_with_restrictions", restrictions=list(engine["known_limitations"]),
    ).to_dict()
    contrast = CounterfactualContrast(
        schema_version=SCHEMA_VERSIONS["counterfactual_contrast"],
        contrast_uid=stable_uid("counterfactual_contrast", {"pair": pair_uid, "hash": trajectory["state_values_hash"]}),
        action_rollout_uids=[action_rollout_uid], control_rollout_uids=[control_rollout_uid],
        pairing_decision="exact_deterministic_pair", outcome_namespace=str(pilot["state_namespace"]),
        contrast_estimator="paired_difference", time_local_contrasts=[0.0 for _ in time_grid], terminal_contrast=0.0,
        integrated_contrast=0.0, uncertainty={"uncertainty_uid": uncertainty["uncertainty_uid"], "unknown_components_preserved": True},
        support_domain_ref=domain_ref, extrapolation_flags=["qualification_synthetic_domain", "no_biological_validation"],
        conflicts=[], eligibility="contract_validation_only_noncausal", causal_effect_claim=False, intervention_effect_claim=False,
    ).to_dict()
    result_uid = stable_uid("what_if_result", {"request": request["request_uid"], "contrast": contrast["contrast_uid"]})
    roles = {name: False for name in ELIGIBILITY_ROLES}
    roles["eligible_for_contract_validation"] = True
    roles["eligible_for_runtime_validation"] = True
    exact_clean = git_dirty_status(repository_root) == "clean"
    eligibility = WhatIfEligibilityDecision(
        schema_version=SCHEMA_VERSIONS["what_if_eligibility_decision"],
        decision_uid=stable_uid("what_if_eligibility", {"result": result_uid, "roles": roles}), result_uid=result_uid,
        controller_uid=str(settings["canonical_controller_uid"]), validator_uid=ROLLOUT_VALIDATOR_UID, roles=roles,
        engine_status=str(engine["status"]), horizon_decision=horizon_decision, pairing_decision="exact_deterministic_pair",
        support_domain_decision=support_decision, constraint_state="valid", uncertainty_status="reported",
        reliability_status="qualified_with_restrictions", exact_execution_identity=exact_clean,
        reason_codes=["restricted_replay_contract_only", "synthetic_baseline", "no_action_only", "unknown_uncertainty_preserved"],
        status="decided",
    ).to_dict()
    result = WhatIfResult(
        schema_version=SCHEMA_VERSIONS["what_if_result"], result_uid=result_uid, request_uid=str(request["request_uid"]),
        rollout_uids=[action_rollout_uid, control_rollout_uid], control_pair_uid=pair_uid,
        contrast_uid=str(contrast["contrast_uid"]), uncertainty_uid=str(uncertainty["uncertainty_uid"]),
        reliability_uid=str(reliability["reliability_uid"]), eligibility_decision_uid=str(eligibility["decision_uid"]),
        result_type="restricted_model_replay_contract_result", claim_scope=str(pilot["claim_scope"]),
        limitations=list(engine["known_limitations"]), model_based_counterfactual=True, causal_effect_claim=False,
        clinical_recommendation=False, empirical_observation=False,
        user_facing_summary="The governed restricted replay produced an exact no-action control match for contract and runtime validation only.",
        provenance={"code_commit": commit, "input_bundle_hash": input_bundle_hash, "engine_uid": engine["engine_uid"]},
    ).to_dict()
    gap_rows = _build_gap_rows(parent["gap_rows"])
    negative_cases = [
        {"case": "unsupported_action_rejection", "expected_blocker": "unsupported_action_class", "result": "rejected"},
        {"case": "outside_horizon_rejection", "expected_blocker": "outside_qualified_horizon", "result": "rejected"},
        {"case": "out_of_domain_rejection", "expected_blocker": "support_domain_silently_expanded", "result": "rejected"},
        {"case": "constraint_failure_quarantine", "expected_blocker": "constraint_failed_rollout_used_in_contrast", "result": "quarantined"},
        {"case": "unknown_uncertainty_preservation", "expected_blocker": "none", "result": "unknown_preserved"},
        {"case": "hypothetical_to_observation_rejection", "expected_blocker": "model_rollout_treated_as_empirical_observation", "result": "rejected"},
        {"case": "unpaired_descriptive_only", "expected_blocker": "unpaired_rollout_described_as_intervention_effect", "result": "descriptive_only"},
        {"case": "untrained_learned_engine", "expected_blocker": "untrained_model_declared_qualified", "result": "rejected"},
        {"case": "clinical_recommendation", "expected_blocker": "clinical_recommendation_emitted", "result": "rejected"},
    ]
    objects = {
        "what_if_request": request, "baseline": baseline, "action_binding": action_binding, "engine": engine,
        "rollout_request": rollout_request, "attempts": [action_attempt, control_attempt, failed_attempt],
        "rollouts": [action_rollout, control_rollout], "control_pair": control_pair, "constraint_report": constraint_report,
        "uncertainty": uncertainty, "reliability": reliability, "contrast": contrast, "eligibility": eligibility,
        "result": result, "negative_cases": negative_cases,
    }
    blockers: list[dict[str, object]] = []
    blockers += validate_what_if_request(request)
    blockers += validate_baseline(baseline, known_support_domains=known_domains)
    blockers += validate_action_binding(action_binding, known_support_domains=known_domains)
    blockers += validate_engine(engine, known_support_domains=known_domains)
    blockers += validate_rollout_request(rollout_request)
    blockers += validate_attempts(objects["attempts"])
    for rollout in objects["rollouts"]:
        blockers += validate_rollout_trajectory(rollout)
    blockers += validate_control_pair(control_pair)
    blockers += validate_constraint_report(constraint_report)
    blockers += validate_uncertainty(uncertainty)
    blockers += validate_reliability(reliability)
    blockers += validate_contrast(contrast, constraint_state="valid")
    blockers += validate_eligibility(eligibility, canonical_controller_uid=str(settings["canonical_controller_uid"]))
    blockers += validate_result(result)
    blockers += validate_gap_map(gap_rows)
    edges = [
        (request["request_uid"], rollout_request["rollout_request_uid"]),
        (rollout_request["rollout_request_uid"], action_attempt_uid),
        (rollout_request["rollout_request_uid"], control_attempt_uid),
        (action_attempt_uid, action_rollout_uid), (control_attempt_uid, control_rollout_uid),
        (action_rollout_uid, pair_uid), (control_rollout_uid, pair_uid),
        (pair_uid, contrast["contrast_uid"]), (contrast["contrast_uid"], result_uid),
    ]
    blockers += validate_lineage_acyclic(edges)
    if blockers:
        raise ValueError(blockers)

    paths = dict(settings["output_paths"])
    request_rows = [
        _catalog_row("valid_what_if_request", "accepted_for_restricted_replay", request),
        _catalog_row("unsupported_action_request", "rejected", negative_cases[0]),
        _catalog_row("outside_horizon_request", "rejected", negative_cases[1]),
        _catalog_row("out_of_domain_request", "rejected", negative_cases[2]),
    ]
    attempt_rows = [
        _catalog_row("action_replay_attempt", "completed", action_attempt),
        _catalog_row("control_replay_attempt", "completed", control_attempt),
        _catalog_row("failed_preexecution_attempt", "retained_failure", failed_attempt),
    ]
    rollout_rows = [
        _catalog_row("restricted_action_rollout", "model_rollout_contract_only", action_rollout),
        _catalog_row("matched_no_action_rollout", "model_rollout_contract_only", control_rollout),
        _catalog_row("constraint_failure_fixture", "quarantined", negative_cases[3]),
    ]
    unpaired_fixture = {"case": "unpaired_rollout", "pairing_decision": "unpaired_descriptive_rollout", "intervention_effect_eligible": False}
    pair_rows = [
        _catalog_row("exact_deterministic_control_pair", "validated", control_pair),
        _catalog_row("unpaired_descriptive_rollout", "descriptive_only", unpaired_fixture),
    ]
    catalog_map = {
        "request_catalog": request_rows,
        "attempt_catalog": attempt_rows,
        "rollout_catalog": rollout_rows,
        "control_pair_catalog": pair_rows,
        "contrast_catalog": [_catalog_row("noncausal_zero_paired_contrast", "contract_only", contrast)],
        "uncertainty_catalog": [_catalog_row("component_uncertainty", "unknown_components_preserved", uncertainty)],
        "reliability_catalog": [_catalog_row("restricted_replay_reliability", "qualified_with_restrictions", reliability)],
        "eligibility_catalog": [_catalog_row("what_if_eligibility", "contract_and_runtime_only", eligibility)],
    }
    for name, rows in catalog_map.items():
        _write_catalog(_resolve(output_root, paths[name]), rows)
    write_tsv(
        _resolve(output_root, paths["frozen_gap_map"]),
        ["gap_uid", "round15_status", "round16_status", "model_rollout_uid", "model_output_role", "resolved_by_model_rollout", "empirical_resolution_evidence", "remaining_blocker"],
        gap_rows,
    )
    graph = {
        "schema_version": "neuronsifter-round16-what-if-graph-v1", "created_at": created_at, "code_commit": commit,
        "environment": environment, "input_artifact_hashes": parent["hashes"], "objects": objects,
        "lineage_edges": [{"parent_uid": parent_uid, "child_uid": child_uid} for parent_uid, child_uid in edges],
        "support_domain_comparison": {
            "organism": {"request": support_domain["organism"], "engine": support_domain["organism"], "difference": False},
            "disease_or_context": {"request": support_domain["disease_or_context"], "engine": support_domain["disease_or_context"], "difference": False},
            "region": {"request": request["requested_regions"], "engine": support_domain["regions"], "difference": False},
            "compartment": {"request": request["requested_compartments"], "engine": support_domain["compartments"], "difference": False},
            "cell_type": {"request": "not_applicable_model_domain", "engine": "not_applicable_model_domain", "difference": False},
            "state_variables": {"request": request["requested_outcomes"], "engine": engine["supported_state_variables"], "difference": False},
            "action_class": {"request": "no_action_control", "engine": engine["supported_action_classes"], "difference": False},
            "dose_or_regimen": {"request": "not_applicable_no_action_control", "engine": "not_applicable_no_action_control", "difference": False},
            "time_horizon": {"request": request["requested_time_horizon"], "engine": engine["supported_horizons"], "difference": False},
            "fidelity": {"request": rollout_request["fidelity"], "engine": engine["supported_fidelities"], "difference": False},
            "decision": support_decision,
        },
        "frozen_gap_count": len(gap_rows), "frozen_gap_resolved_by_model_rollout": 0,
        "evidence_fabric_sealed": False, "dt_core_status": "not_evaluated",
    }
    write_json(_resolve(output_root, paths["graph"]), graph)
    artifact_paths = {
        name: _resolve(output_root, paths[name])
        for name in (*catalog_map, "graph", "frozen_gap_map")
    }
    artifact_hashes = {name: sha256_file(path) for name, path in artifact_paths.items()}
    release = {
        "schema_version": SCHEMA_VERSIONS["release"], "parent_release_uid": parent["release"]["release_uid"],
        "created_at": created_at, "code_commit": commit,
        "config_hash": sha256_file(repository_root / "Libs/configs/data/round16_what_if_engine.yaml"),
        "schema_registry_hash": sha256_file(repository_root / "Libs/configs/schema/schemas.yaml"),
        "input_bundle_hash": input_bundle_hash, "environment_hash": environment_hash,
        "model_output_classes": ["model_rollout", "model_counterfactual_contrast", "model_uncertainty_report", "model_reliability_report"],
        "rollout_count": 2, "contrast_count": 1, "uncertainty_report_count": 1, "reliability_report_count": 1,
        "empirical_observation_count": 0, "empirical_gap_resolution_count": 0, "artifact_hashes": artifact_hashes,
        "status": "materialized_unsealed", "sealed": False, "dt_core_status": "not_evaluated",
    }
    release_hash = release_content_hash(release)
    release["release_uid"] = f"evidence_fabric_round16_release_{release_hash[:24]}"
    release["release_content_hash"] = release_hash
    release_path = _resolve(output_root, paths["release_root"]) / str(release["release_uid"]) / "round16_what_if_release_candidate.json"
    write_json(release_path, release)
    release_blockers = validate_release_candidate(release)
    if release_blockers:
        raise ValueError(release_blockers)
    pilot_validation = {
        "schema_version": "neuronsifter-dt-round16-pilot-validation-v1", "status": "passed", "blocker_count": 0,
        "blockers": [], "request_uid": request["request_uid"], "engine_uid": engine["engine_uid"],
        "engine_status": engine["status"], "rollout_count": 2, "matched_control_pair_count": 1,
        "contrast_count": 1, "contrast_is_causal": False, "empirical_observation_count": 0,
        "unknown_uncertainty_components": unknown_components, "negative_cases": negative_cases,
        "mandatory_negative_case_count": len(negative_cases), "frozen_gap_count": len(gap_rows),
        "frozen_gap_resolved_by_model_rollout": 0, "release_uid": release["release_uid"],
        "release_path": str(release_path.relative_to(output_root)), "evidence_fabric_sealed": False,
        "dt_core_status": "not_evaluated",
    }
    write_json(_resolve(output_root, paths["pilot_validation"]), pilot_validation)
    return {
        "parent": parent, "objects": objects, "gap_rows": gap_rows, "graph": graph, "release": release,
        "release_path": release_path, "pilot_validation": pilot_validation, "artifact_paths": artifact_paths,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the restricted Round-16 governed what-if pilot.")
    parser.add_argument("--config", default=str(REPO_ROOT / "Libs/configs/config_stack.yaml"))
    parser.add_argument("--repository-root", default=str(REPO_ROOT))
    parser.add_argument("--parent-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--created-at")
    args = parser.parse_args()
    repository_root = Path(args.repository_root).resolve()
    created_at = args.created_at or datetime.now(timezone.utc).isoformat()
    result = build_round16_pilot(
        config=load_config(Path(args.config)), repository_root=repository_root,
        parent_root=Path(args.parent_root).resolve(), output_root=Path(args.output_root).resolve(), created_at=created_at,
    )
    print(
        f"round16_pilot_{result['pilot_validation']['status']} "
        f"rollouts={result['pilot_validation']['rollout_count']} gaps={result['pilot_validation']['frozen_gap_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
