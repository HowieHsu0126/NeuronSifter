"""Build the frozen Operational Twin training protocol preflight artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Libs.utils.evaluation.operational_twin_training_protocol import (  # noqa: E402
    CANDIDATE_NEGATIVE_TERMINAL,
    build_candidate_assignment_manifest,
    build_candidate_lineage_manifest,
    build_candidate_preflight_authority,
    build_assignment_manifest,
    build_preflight_authority,
    build_resource_receipt,
    read_repo_bytes,
    sha256_file,
    sha256_text,
    validate_candidate_artifacts,
    validate_candidate_protocol_config,
    validate_open_data_binding,
    validate_protocol_config,
)
from Libs.scripts.builders.build_operational_twin_development_training_commands import (  # noqa: E402
    build_candidate_commands,
)
from Libs.utils.exp_utils import load_config_stack  # noqa: E402
from Libs.utils.evaluation.manuscript_native_protocol import publish_package  # noqa: E402
from Libs.utils.evaluation.operational_twin_candidate_execution import (  # noqa: E402
    validate_candidate_execution_interface_repair,
    validate_candidate_receipt_semantics_amendment,
)
from Libs.utils.evaluation.operational_twin_candidate_evidence import (  # noqa: E402
    validate_scheduler_accounting_fixture,
)

from Libs.utils.evaluation.operational_twin_candidate_retry_authority import (  # noqa: E402
    POSITIVE as RETRY_AUTHORITY_POSITIVE,
    adjudicate_retry_authority,
    derive_positive_successor,
    materialized_positive_audit,
    render_positive_successor_report,
    render_retry_authority_report,
    validate_positive_successor_package,
)
from Libs.utils.evaluation.operational_twin_successor_lineage_protocol import (  # noqa: E402
    INVALID_TERMINAL as SUCCESSOR_INVALID_TERMINAL,
    build_successor_package,
    publication_paths as successor_publication_paths,
)

def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False)


def _probe(command: list[str], job_id: str, job_name: str, partition: str, node: str, cpus: int, mem: str) -> dict[str, object]:
    completed = _run(command)
    return {
        "job_id": job_id,
        "job_name": job_name,
        "partition": partition,
        "node": node,
        "cpus_per_task": cpus,
        "memory": mem,
        "command": command,
        "exit_status": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "sacct_evidence": "",
    }


def _write_report(path: Path, preflight: dict[str, object], receipt: dict[str, object]) -> None:
    lines = [
        "# Operational Twin Training Protocol Preflight",
        "",
        f"- Status: `{preflight['status']}`",
        f"- Terminal: `{preflight['terminal_decision']}`",
        f"- Protocol UID: `{preflight['protocol_uid']}`",
        f"- Methods: `{len(preflight['method_ids'])}`",
        f"- Seeds: `{len(preflight['seeds'])}`",
        f"- Assignments: `{preflight['assignment_count']}`",
        f"- Open-data terminal: `{preflight['open_data_terminal']}`",
        f"- Sufficient training: `{preflight['sufficient_training']}`",
        f"- Twin qualification pass: `{preflight['twin_qualification_pass']}`",
        f"- Formal qualification: `{preflight['formal_qualification']}`",
        f"- Training executed: `{preflight['training_executed']}`",
        f"- Calibration fitting executed: `{preflight['calibration_fitting_executed']}`",
        f"- Checkpoint created: `{preflight['checkpoint_created']}`",
        "",
        "## Resource Probes",
        "",
    ]
    for probe_id, probe in dict(receipt["probes"]).items():
        lines.append(
            f"- `{probe_id}`: job `{probe['job_id']}` on `{probe['node']}`/"
            f"`{probe['partition']}`, exit `{probe['exit_status']}`"
        )
    lines.extend(
        [
            "",
            "All sealed/test/protected/six-context/Tool/MCP/AgentTrace/Agent/Memory/Co-Evolution counters remain zero.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _candidate_report(preflight: dict[str, object]) -> str:
    resource = dict(preflight["resource_execution_gate"])
    lines = [
        "# Operational Twin Candidate-Lineage Protocol Preflight",
        "",
        f"- Status: `{preflight['status']}`",
        f"- Terminal: `{preflight['terminal_decision']}`",
        f"- Authoritative date: `{preflight['authoritative_date']}`",
        f"- Candidate lineage UID: `{preflight['candidate_lineage_uid']}`",
        f"- Protocol UID: `{preflight['protocol_uid']}`",
        f"- Candidate manifest hash: `{preflight['candidate_manifest_hash']}`",
        f"- Assignment manifest hash: `{preflight['assignment_manifest_hash']}`",
        f"- Command catalog hash: `{preflight['command_catalog_hash']}`",
        f"- Assignments: `{preflight['assignment_count']}`",
        f"- Frozen attempt slots: `{preflight['attempt_slot_count']}`",
        "",
        "## Execution Boundary",
        "",
        f"- Fresh `sinfo` required before dispatch: `{resource['fresh_sinfo_required_before_dispatch']}`",
        f"- New resource/environment receipt required before dispatch: `{resource['new_resource_environment_receipt_required_before_dispatch']}`",
        f"- Resource receipt bound in this round: `{resource['resource_receipt_bound_in_this_round']}`",
        "- Trainable placement: `gpu03/gpu1`, profile `gpu_single`, non-exclusive.",
        "- No-fit placement: `cpu04/cpu`, profile `cpu_light`, non-exclusive.",
        "- Training executed: `false`",
        "- Calibration fitting executed: `false`",
        "- Checkpoint created: `false`",
        "- Sufficient training: `false`",
        "- Twin qualification pass: `false`",
        "- Formal qualification: `false`",
        "- Pre-sealed freeze created: `false`",
        "",
        "The historical candidate remains `OPERATIONAL_TWIN_TRAINING_EVIDENCE_INVALID`; its scientific Twin result remains `NOT_RUN`.",
        "All sealed/test-outcome/protected/six-context and downstream access counters remain zero and closed.",
    ]
    return "\n".join(lines) + "\n"


def _content_identity(path: str, content: bytes) -> dict[str, object]:
    return {
        "path": path,
        "sha256": sha256_text(content.decode("utf-8")),
        "size_bytes": len(content),
    }


def _amendment_report(audit: Mapping[str, object]) -> str:
    lines = [
        "# Operational Twin Resource-Receipt Semantics",
        "",
        f"- Status: `{audit['status']}`",
        f"- Terminal: `{audit['terminal_decision']}`",
        f"- Candidate lineage UID: `{audit['candidate_lineage_uid']}`",
        f"- Protocol UID: `{audit['protocol_uid']}`",
        f"- Preflight UID: `{audit['preflight_uid']}`",
        f"- Receipt schema: `{dict(audit['schema_matrix'])['resource_receipt']}`",
        f"- Authority schema: `{dict(audit['schema_matrix'])['execution_authority']}`",
        "",
        "## Scheduler Mapping",
        "",
    ]
    for normalized, raw in dict(audit["raw_field_mapping"]).items():
        lines.append(f"- `{normalized}` <- `{raw}`")
    lines.extend(
        [
            "",
            "## Preserved Boundaries",
            "",
            f"- Predecessor files preserved: `{audit['predecessor_file_count']}`",
            f"- Assignments: `{audit['assignment_count']}`",
            f"- Attempt slots: `{audit['attempt_count']}`",
            "- Candidate execution authority created: `false`",
            "- Candidate resource receipt created: `false`",
            "- Candidate training/calibration/checkpoint execution: `false`",
            "- Scientific Twin result: `NOT_RUN`",
            "- All sealed/test/protected/six-context/downstream access counters: `0`",
        ]
    )
    return "\n".join(lines) + "\n"


def _interface_repair_report(audit: Mapping[str, object]) -> str:
    return "\n".join(
        [
            "# Operational Twin Candidate Execution Interfaces",
            "",
            f"- Status: `{audit['status']}`",
            f"- Terminal: `{audit['terminal_decision']}`",
            f"- Candidate lineage UID: `{audit['candidate_lineage_uid']}`",
            f"- Protocol UID: `{audit['protocol_uid']}`",
            f"- Preflight UID: `{audit['preflight_uid']}`",
            f"- Predecessor files preserved: `{audit['predecessor_file_count']}`",
            f"- Predecessor pre/post identity match: `{str(audit['predecessor_pre_post_match']).lower()}`",
            f"- Scheduler probe logs retained: `{len(audit['scheduler_probe_logs'])}`",
            f"- Candidate execution side effects: `{audit['side_effect_count']}`",
            f"- Evaluator authority interface: `{audit['interfaces']['calibration_authority_evaluation']}`",
            f"- Array resource identity interface: `{audit['interfaces']['array_element_resource_identity']}`",
            f"- Terminal accounting interface: `{audit['interfaces']['terminal_scheduler_mapping']}`",
            "",
            "## Preserved Boundaries",
            "",
            "- Candidate dispatch authorized: `false`",
            "- Candidate training/calibration/checkpoint execution: `false`",
            "- Recovery eligibility or aggregation created: `false`",
            "- Scientific Twin result: `NOT_RUN`",
            "- All sealed/test/protected/six-context/downstream access counters: `0`",
            "",
        ]
    )


def _artifact_identity(root: Path, path: Path) -> dict[str, object]:
    resolved = path.resolve()
    return {
        "path": resolved.relative_to(root.resolve()).as_posix(),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _validate_interface_repair_evidence(
    *,
    root: Path,
    protocol: Mapping[str, object],
    resource_fixture: Mapping[str, object],
    terminal_fixture: Mapping[str, object],
    pre_change_identity_manifest_path: Path,
) -> tuple[list[dict[str, str]], dict[str, object], list[dict[str, object]]]:
    blockers: list[dict[str, str]] = []
    try:
        manifest = json.loads(
            pre_change_identity_manifest_path.read_text(encoding="utf-8")
        )
        if not isinstance(manifest, dict):
            raise ValueError("pre-change identity manifest must contain an object")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return (
            [
                {
                    "code": "candidate_interface_repair_prechange_manifest_invalid",
                    "message": str(error),
                }
            ],
            {},
            [],
        )

    required_keys = {
        "schema_version",
        "captured_at",
        "source_commit",
        "candidate_lineage_uid",
        "predecessor_protocol_artifacts",
        "round_46_evidence",
        "scheduler_probe_logs",
        "successor_paths_absent",
        "operational_boundary",
    }
    if set(manifest) != required_keys:
        blockers.append(
            {
                "code": "candidate_interface_repair_prechange_manifest_invalid",
                "message": "pre-change identity manifest keyset is not exact",
            }
        )

    repair_basis = dict(
        dict(protocol["supersession_contract"])["execution_interface_repair_basis"]
    )
    predecessor_generations = dict(
        dict(protocol["supersession_contract"])[
            "resource_receipt_amendment_basis"
        ]["predecessor_generations"]
    )
    current_predecessors: list[dict[str, object]] = []
    for generation_name, generation_value in predecessor_generations.items():
        generation = dict(generation_value)
        for role, identity_value in dict(generation["artifacts"]).items():
            identity = dict(identity_value)
            path = root / str(identity["path"])
            try:
                current = _artifact_identity(root, path)
            except (OSError, ValueError) as error:
                blockers.append(
                    {
                        "code": "candidate_interface_repair_predecessor_unavailable",
                        "message": f"{generation_name}/{role}: {error}",
                    }
                )
                continue
            current_predecessors.append(
                {
                    "generation": generation_name,
                    "role": role,
                    **current,
                }
            )
            if current["sha256"] != identity["sha256"]:
                blockers.append(
                    {
                        "code": "candidate_interface_repair_predecessor_drift",
                        "message": f"{generation_name}/{role} is not byte-identical",
                    }
                )
    current_predecessors.sort(key=lambda row: (str(row["generation"]), str(row["role"])))
    declared_predecessors = manifest.get("predecessor_protocol_artifacts")
    if declared_predecessors != current_predecessors:
        blockers.append(
            {
                "code": "candidate_interface_repair_prechange_manifest_drift",
                "message": "predecessor protocol identities differ from the pre-change manifest",
            }
        )

    current_round_46_evidence: dict[str, dict[str, object]] = {}
    for role in (
        "approved_round_46_review_record",
        "implementation_notes",
        "blocker_summary",
    ):
        configured = dict(repair_basis[role])
        path = root / str(configured["path"])
        try:
            current = _artifact_identity(root, path)
        except (OSError, ValueError) as error:
            blockers.append(
                {
                    "code": "candidate_interface_repair_round_46_evidence_unavailable",
                    "message": f"{role}: {error}",
                }
            )
            continue
        current_round_46_evidence[role] = {
            **current,
            "git_object_id": configured["git_object_id"],
        }
        if current["sha256"] != configured["sha256"]:
            blockers.append(
                {
                    "code": "candidate_interface_repair_round_46_evidence_drift",
                    "message": f"{role} is not byte-identical",
                }
            )
    if manifest.get("round_46_evidence") != current_round_46_evidence:
        blockers.append(
            {
                "code": "candidate_interface_repair_prechange_manifest_drift",
                "message": "Round 46 evidence differs from the pre-change manifest",
            }
        )

    paths = dict(protocol["output_paths"])
    expected_successor_paths = sorted(
        str(paths[key])
        for key in (
            "candidate_lineage_manifest",
            "assignment_manifest",
            "assignment_commands",
            "preflight_authority",
            "preflight_report",
            "interface_repair_audit",
            "interface_repair_report",
        )
    )
    if manifest.get("successor_paths_absent") != expected_successor_paths:
        blockers.append(
            {
                "code": "candidate_interface_repair_prechange_manifest_drift",
                "message": "successor path set differs from the pre-change manifest",
            }
        )
    existing_successors = [
        path for path in expected_successor_paths if os.path.lexists(root / path)
    ]
    if existing_successors:
        blockers.append(
            {
                "code": "candidate_interface_repair_successor_collision",
                "message": "successor paths were not absent before publication: "
                + ", ".join(existing_successors),
            }
        )

    if (
        manifest.get("schema_version")
        != "neuronsifter-operational-twin-candidate-interface-prechange-v1"
        or manifest.get("candidate_lineage_uid")
        != protocol["candidate_lineage_uid"]
        or manifest.get("operational_boundary")
        != repair_basis["operational_boundary"]
    ):
        blockers.append(
            {
                "code": "candidate_interface_repair_prechange_manifest_invalid",
                "message": "pre-change identity or operational boundary drifted",
            }
        )
    source_commit = str(manifest.get("source_commit") or "")
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        blockers.append(
            {
                "code": "candidate_interface_repair_prechange_manifest_invalid",
                "message": "pre-change source commit must be one full Git object id",
            }
        )
    elif (root / ".git").exists():
        completed = subprocess.run(
            ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            blockers.append(
                {
                    "code": "candidate_interface_repair_prechange_source_invalid",
                    "message": "pre-change source commit is not an ancestor of HEAD",
                }
            )

    source_job_id = str(terminal_fixture.get("source_job_id") or "")
    expected_log_paths = [
        (
            root
            / "orchestrator/rounds"
            / str(dict(protocol["execution_authority_contract"])["required_review_round_id"])
            / f"array_element_probe_{source_job_id}_{index}.log"
        )
        for index in terminal_fixture.get("command_indices", [])
    ]
    current_logs: list[dict[str, object]] = []
    terminal_rows = {
        int(str(row["job_id"]).rsplit("_", 1)[1]): dict(row)
        for row in terminal_fixture.get("expected_terminal_rows", [])
    }
    for index, path in zip(terminal_fixture.get("command_indices", []), expected_log_paths, strict=True):
        try:
            identity = _artifact_identity(root, path)
            content = path.read_text(encoding="utf-8")
        except (OSError, ValueError) as error:
            blockers.append(
                {
                    "code": "candidate_interface_repair_probe_log_unavailable",
                    "message": str(error),
                }
            )
            continue
        expected_runtime_job_id = str(terminal_rows[int(index)]["job_id_raw"])
        required_fragments = (
            f"SLURM_JOB_ID={expected_runtime_job_id}",
            f"SLURM_ARRAY_JOB_ID={source_job_id}",
            f"SLURM_ARRAY_TASK_ID={index}",
            f"RESOURCE_QUERY_JOB_ID={source_job_id}_{index}",
            f"JobId={expected_runtime_job_id} ArrayJobId={source_job_id} ArrayTaskId={index}",
        )
        if any(fragment not in content for fragment in required_fragments):
            blockers.append(
                {
                    "code": "candidate_interface_repair_probe_log_invalid",
                    "message": f"scheduler probe log {path.name} lacks exact runtime identity evidence",
                }
            )
        current_logs.append(identity)
    if manifest.get("scheduler_probe_logs") != current_logs:
        blockers.append(
            {
                "code": "candidate_interface_repair_prechange_manifest_drift",
                "message": "scheduler probe-log identities differ from the pre-change manifest",
            }
        )
    if str(resource_fixture.get("source_job_id")) != source_job_id:
        blockers.append(
            {
                "code": "candidate_interface_repair_probe_fixture_mismatch",
                "message": "resource and terminal fixtures do not share one array job",
            }
        )

    preserved_identities = [
        *current_predecessors,
        *current_round_46_evidence.values(),
        *current_logs,
        _artifact_identity(root, pre_change_identity_manifest_path),
    ]
    return blockers, manifest, preserved_identities


def _build_candidate_receipt_amendment(
    config: dict[str, object],
    authoritative_date: str,
    *,
    cpu_probe_receipt_path: Path,
    gpu_probe_receipt_path: Path,
    validation_evidence_path: Path,
) -> dict[str, object]:
    blockers = validate_candidate_receipt_semantics_amendment(config, REPO_ROOT)
    protocol = dict(config["operational_twin_candidate_lineage_protocol"])
    if protocol.get("authoritative_date") != authoritative_date:
        blockers.append(
            {
                "code": "candidate_authoritative_date_mismatch",
                "message": "CLI authoritative date must match the candidate-lineage protocol.",
            }
        )
    evidence_paths = {
        "cpu_light": cpu_probe_receipt_path,
        "gpu_single": gpu_probe_receipt_path,
    }
    live_probe_receipts: dict[str, object] = {}
    for profile, path in evidence_paths.items():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") != "PASS" or payload.get("side_effect_count") != 0:
                raise ValueError("live probe is not a zero-side-effect PASS")
            live_probe_receipts[profile] = {
                "identity": {
                    "path": path.as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                },
                "receipt": payload,
            }
        except (OSError, ValueError, json.JSONDecodeError) as error:
            blockers.append(
                {
                    "code": "candidate_live_probe_receipt_invalid",
                    "message": f"{profile}: {error}",
                }
            )
    try:
        validation_evidence = json.loads(
            validation_evidence_path.read_text(encoding="utf-8")
        )
        if validation_evidence.get("status") != "PASS":
            raise ValueError("validation evidence is not PASS")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        blockers.append(
            {
                "code": "candidate_amendment_validation_evidence_invalid",
                "message": str(error),
            }
        )
        validation_evidence = {}
    if blockers:
        return {
            "status": "BLOCKED",
            "terminal_decision": "OPERATIONAL_TWIN_RESOURCE_RECEIPT_SEMANTICS_INVALID",
            "blockers": blockers,
        }

    candidate_manifest = build_candidate_lineage_manifest(config)
    assignment_manifest = build_candidate_assignment_manifest(config, candidate_manifest)
    command_catalog = build_candidate_commands(
        config,
        candidate_manifest,
        assignment_manifest,
    )
    preflight = build_candidate_preflight_authority(
        config,
        REPO_ROOT,
        candidate_manifest,
        assignment_manifest,
        command_catalog,
    )
    blockers = validate_candidate_artifacts(
        config,
        REPO_ROOT,
        candidate_manifest,
        assignment_manifest,
        command_catalog,
        preflight,
        require_attempt_roots_absent=True,
    )
    if blockers:
        return {
            "status": "BLOCKED",
            "terminal_decision": "OPERATIONAL_TWIN_RESOURCE_RECEIPT_SEMANTICS_INVALID",
            "blockers": blockers,
        }

    paths = dict(protocol["output_paths"])
    candidate_bytes = _json_bytes(candidate_manifest)
    assignment_bytes = _json_bytes(assignment_manifest)
    command_bytes = _json_bytes(command_catalog)
    preflight_bytes = _json_bytes(preflight)
    preflight_report_bytes = _candidate_report(preflight).encode("utf-8")
    successor_content = {
        "candidate_lineage_manifest": (
            str(paths["candidate_lineage_manifest"]),
            candidate_bytes,
        ),
        "assignment_manifest": (
            str(paths["assignment_manifest"]),
            assignment_bytes,
        ),
        "assignment_commands": (
            str(paths["assignment_commands"]),
            command_bytes,
        ),
        "preflight_authority": (
            str(paths["preflight_authority"]),
            preflight_bytes,
        ),
        "preflight_report": (
            str(paths["preflight_report"]),
            preflight_report_bytes,
        ),
    }
    amendment = dict(protocol["supersession_contract"])[
        "resource_receipt_amendment_basis"
    ]
    predecessor_files = [
        {
            "generation": generation_name,
            "role": role,
            **dict(identity),
        }
        for generation_name, generation in dict(
            amendment["predecessor_generations"]
        ).items()
        for role, identity in dict(generation["artifacts"]).items()
    ]
    audit: dict[str, object] = {
        "schema_version": "neuronsifter-operational-twin-resource-receipt-semantics-audit-v1",
        "status": "PASS",
        "terminal_decision": "OPERATIONAL_TWIN_RESOURCE_RECEIPT_SEMANTICS_FROZEN",
        "amendment_date": dict(protocol["scheduler_evidence_contract"])[
            "amendment_date"
        ],
        "candidate_lineage_uid": protocol["candidate_lineage_uid"],
        "predecessor_protocol_uid": "operational_twin_candidate_protocol_c7e36201c1534c954f888337",
        "protocol_uid": protocol["protocol_uid"],
        "predecessor_preflight_uid": "operational_twin_candidate_lineage_preflight_f1aa5c5d79c7cb5e9e02b447",
        "preflight_uid": preflight["preflight_uid"],
        "schema_matrix": {
            "protocol": protocol["schema_version"],
            "candidate_manifest": candidate_manifest["schema_version"],
            "assignment_manifest": assignment_manifest["schema_version"],
            "command_catalog": command_catalog["schema_version"],
            "preflight": preflight["schema_version"],
            "resource_receipt": dict(protocol["execution_authority_contract"])[
                "resource_receipt_schema_version"
            ],
            "execution_authority": dict(protocol["execution_authority_contract"])[
                "authority_schema_version"
            ],
            "candidate_execution": dict(
                config["operational_twin_candidate_execution"]
            )["schema_version"],
        },
        "raw_field_mapping": dict(protocol["scheduler_evidence_contract"])[
            "field_mapping"
        ],
        "forbidden_legacy_fields": dict(protocol["scheduler_evidence_contract"])[
            "forbidden_legacy_fields"
        ],
        "predecessor_file_count": len(predecessor_files),
        "predecessor_files": predecessor_files,
        "successor_artifacts": {
            role: _content_identity(path, content)
            for role, (path, content) in successor_content.items()
        },
        "retained_fixtures": {
            profile: {
                **dict(binding),
                "sha256": sha256_file(REPO_ROOT / str(binding["path"])),
            }
            for profile, binding in dict(
                dict(protocol["scheduler_evidence_contract"])["retained_fixtures"]
            ).items()
        },
        "live_probe_receipts": live_probe_receipts,
        "validation_evidence": validation_evidence,
        "assignment_count": assignment_manifest["assignment_count"],
        "attempt_count": assignment_manifest["attempt_slot_count"],
        "candidate_manifest_hash": candidate_manifest["candidate_manifest_hash"],
        "assignment_manifest_hash": assignment_manifest[
            "assignment_manifest_hash"
        ],
        "command_catalog_hash": command_catalog["command_catalog_hash"],
        "materialized_attempt_root_count": 0,
        "materialized_resource_receipt_count": 0,
        "materialized_execution_authority_count": 0,
        "materialized_execution_output_count": 0,
        "training_executed": False,
        "calibration_fitting_executed": False,
        "checkpoint_created": False,
        "candidate_execution_authorized": False,
        "scientific_twin_result": "NOT_RUN",
        "access_counters": protocol["access_counters"],
        "closed_flags": protocol["closed_flags"],
    }
    audit_bytes = _json_bytes(audit)
    report_bytes = _amendment_report(audit).encode("utf-8")
    publication = _publish_candidate_artifacts(
        [
            *((REPO_ROOT / path, content) for path, content in successor_content.values()),
            (REPO_ROOT / str(paths["amendment_audit"]), audit_bytes),
            (REPO_ROOT / str(paths["amendment_report"]), report_bytes),
        ],
        protocol_root=REPO_ROOT / str(paths["protocol_root"]),
    )
    return {
        "status": "PASS",
        "terminal_decision": audit["terminal_decision"],
        "candidate_lineage_uid": audit["candidate_lineage_uid"],
        "protocol_uid": audit["protocol_uid"],
        "preflight_uid": audit["preflight_uid"],
        "candidate_manifest_hash": audit["candidate_manifest_hash"],
        "assignment_manifest_hash": audit["assignment_manifest_hash"],
        "command_catalog_hash": audit["command_catalog_hash"],
        "publication": publication,
    }



def _build_candidate_execution_interface_repair(
    config: dict[str, object],
    authoritative_date: str,
    *,
    array_resource_fixture_path: Path,
    array_terminal_fixture_path: Path,
    pre_change_identity_manifest_path: Path,
    validation_evidence_path: Path,
) -> dict[str, object]:
    protocol = dict(config["operational_twin_candidate_lineage_protocol"])
    execution = dict(config["operational_twin_candidate_execution"])
    repair = dict(execution["interface_repair_contract"])
    blockers = validate_candidate_execution_interface_repair(
        config,
        REPO_ROOT,
        require_materialized_artifacts=False,
    )
    if protocol.get("authoritative_date") != authoritative_date:
        blockers.append(
            {
                "code": "candidate_authoritative_date_mismatch",
                "message": "CLI authoritative date must match the candidate-lineage protocol.",
            }
        )
    source_bindings = dict(protocol["source_bindings"])
    configured_fixtures = {
        "array_element_resource": (
            REPO_ROOT / str(source_bindings["array_element_resource_fixture"])
        ).resolve(),
        "array_terminal_accounting": (
            REPO_ROOT / str(source_bindings["array_terminal_accounting_fixture"])
        ).resolve(),
    }
    supplied_fixtures = {
        "array_element_resource": array_resource_fixture_path.resolve(),
        "array_terminal_accounting": array_terminal_fixture_path.resolve(),
    }
    for role, expected_path in configured_fixtures.items():
        if supplied_fixtures[role] != expected_path:
            blockers.append(
                {
                    "code": "candidate_interface_repair_fixture_path_mismatch",
                    "message": f"{role} fixture does not match the configured source binding.",
                }
            )
    try:
        resource_fixture = json.loads(
            array_resource_fixture_path.read_text(encoding="utf-8")
        )
        terminal_fixture = json.loads(
            array_terminal_fixture_path.read_text(encoding="utf-8")
        )
        blockers.extend(validate_scheduler_accounting_fixture(terminal_fixture))
    except (OSError, json.JSONDecodeError) as error:
        blockers.append(
            {
                "code": "candidate_interface_repair_fixture_unavailable",
                "message": str(error),
            }
        )
        resource_fixture = {}
        terminal_fixture = {}
    try:
        validation_evidence = json.loads(
            validation_evidence_path.read_text(encoding="utf-8")
        )
        if (
            validation_evidence.get("status") != "PASS"
            or validation_evidence.get("terminal_decision")
            != repair["positive_terminal"]
            or validation_evidence.get("side_effect_count") != 0
        ):
            raise ValueError(
                "validation evidence is not a zero-side-effect interface-repair PASS"
            )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        blockers.append(
            {
                "code": "candidate_interface_repair_validation_evidence_invalid",
                "message": str(error),
            }
        )
        validation_evidence = {}
    pre_change_manifest: dict[str, object] = {}
    preserved_identities: list[dict[str, object]] = []
    if not blockers:
        (
            evidence_blockers,
            pre_change_manifest,
            preserved_identities,
        ) = _validate_interface_repair_evidence(
            root=REPO_ROOT,
            protocol=protocol,
            resource_fixture=resource_fixture,
            terminal_fixture=terminal_fixture,
            pre_change_identity_manifest_path=pre_change_identity_manifest_path,
        )
        blockers.extend(evidence_blockers)
    if blockers:
        return {
            "status": "BLOCKED",
            "terminal_decision": repair["negative_terminal"],
            "blockers": blockers,
        }

    candidate_manifest = build_candidate_lineage_manifest(config)
    assignment_manifest = build_candidate_assignment_manifest(
        config,
        candidate_manifest,
    )
    command_catalog = build_candidate_commands(
        config,
        candidate_manifest,
        assignment_manifest,
    )
    preflight = build_candidate_preflight_authority(
        config,
        REPO_ROOT,
        candidate_manifest,
        assignment_manifest,
        command_catalog,
    )
    blockers = validate_candidate_artifacts(
        config,
        REPO_ROOT,
        candidate_manifest,
        assignment_manifest,
        command_catalog,
        preflight,
        require_attempt_roots_absent=False,
    )
    if blockers:
        return {
            "status": "BLOCKED",
            "terminal_decision": repair["negative_terminal"],
            "blockers": blockers,
        }

    paths = dict(protocol["output_paths"])
    successor_content = {
        "candidate_lineage_manifest": (
            str(paths["candidate_lineage_manifest"]),
            _json_bytes(candidate_manifest),
        ),
        "assignment_manifest": (
            str(paths["assignment_manifest"]),
            _json_bytes(assignment_manifest),
        ),
        "assignment_commands": (
            str(paths["assignment_commands"]),
            _json_bytes(command_catalog),
        ),
        "preflight_authority": (
            str(paths["preflight_authority"]),
            _json_bytes(preflight),
        ),
        "preflight_report": (
            str(paths["preflight_report"]),
            _candidate_report(preflight).encode("utf-8"),
        ),
    }
    predecessor_generations = dict(
        dict(protocol["supersession_contract"])[
            "resource_receipt_amendment_basis"
        ]["predecessor_generations"]
    )
    predecessor_files = [
        {"generation": generation_name, "role": role, **dict(identity)}
        for generation_name, generation in predecessor_generations.items()
        for role, identity in dict(generation["artifacts"]).items()
    ]
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    audit: dict[str, object] = {
        "schema_version": "neuronsifter-operational-twin-candidate-execution-interface-repair-audit-v1",
        "status": "PASS",
        "terminal_decision": repair["positive_terminal"],
        "authoritative_date": authoritative_date,
        "source_commit": source_commit,
        "candidate_lineage_uid": protocol["candidate_lineage_uid"],
        "protocol_uid": protocol["protocol_uid"],
        "preflight_uid": preflight["preflight_uid"],
        "candidate_manifest_hash": candidate_manifest["candidate_manifest_hash"],
        "assignment_manifest_hash": assignment_manifest[
            "assignment_manifest_hash"
        ],
        "command_catalog_hash": command_catalog["command_catalog_hash"],
        "interfaces": {
            "calibration_authority_evaluation": "PASS",
            "array_element_resource_identity": "PASS",
            "terminal_scheduler_mapping": "PASS",
        },
        "scheduler_fixtures": {
            "array_element_resource": {
                "path": array_resource_fixture_path.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256_file(array_resource_fixture_path),
                "source_job_id": resource_fixture.get("source_job_id"),
                "captured_at": resource_fixture.get("captured_at"),
                "raw_stdout_sha256": resource_fixture.get(
                    "resource_stdout_sha256"
                ),
            },
            "array_terminal_accounting": {
                "path": array_terminal_fixture_path.relative_to(REPO_ROOT).as_posix(),
                "sha256": sha256_file(array_terminal_fixture_path),
                "source_job_id": terminal_fixture.get("source_job_id"),
                "captured_at": terminal_fixture.get("captured_at"),
                "raw_stdout_sha256": terminal_fixture.get("stdout_sha256"),
            },
        },
        "validation_evidence": {
            "path": validation_evidence_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": sha256_file(validation_evidence_path),
            "payload": validation_evidence,
        },
        "pre_change_identity_manifest": {
            **_artifact_identity(REPO_ROOT, pre_change_identity_manifest_path),
            "payload": pre_change_manifest,
        },
        "scheduler_probe_logs": pre_change_manifest["scheduler_probe_logs"],
        "scheduler_timestamp_handling": {
            "fixture_capture_timezone": "UTC",
            "raw_slurm_timestamps_preserved_verbatim": True,
            "raw_slurm_timestamps_used_for_freshness": False,
        },
        "predecessor_file_count": len(predecessor_files),
        "predecessor_files": predecessor_files,
        "predecessor_pre_post_match": True,
        "successor_artifacts": {
            role: _content_identity(path, content)
            for role, (path, content) in successor_content.items()
        },
        "candidate_dispatch_authorized": False,
        "materialized_candidate_attempt_count": 0,
        "materialized_candidate_execution_artifact_count": 0,
        "training_executed": False,
        "calibration_fitting_executed": False,
        "checkpoint_created": False,
        "recovery_eligibility_created": False,
        "execution_catalog_created": False,
        "sufficiency_adjudicated": False,
        "scientific_twin_result": "NOT_RUN",
        "side_effect_count": 0,
        "access_counters": protocol["access_counters"],
        "closed_flags": protocol["closed_flags"],
    }
    publication = _publish_candidate_artifacts(
        [
            *((REPO_ROOT / path, content) for path, content in successor_content.values()),
            (
                REPO_ROOT / str(paths["interface_repair_audit"]),
                _json_bytes(audit),
            ),
            (
                REPO_ROOT / str(paths["interface_repair_report"]),
                _interface_repair_report(audit).encode("utf-8"),
            ),
        ],
        protocol_root=REPO_ROOT / str(paths["protocol_root"]),
        preserved_identities=preserved_identities,
    )
    return {
        "status": "PASS",
        "terminal_decision": audit["terminal_decision"],
        "candidate_lineage_uid": audit["candidate_lineage_uid"],
        "protocol_uid": audit["protocol_uid"],
        "preflight_uid": audit["preflight_uid"],
        "candidate_manifest_hash": audit["candidate_manifest_hash"],
        "assignment_manifest_hash": audit["assignment_manifest_hash"],
        "command_catalog_hash": audit["command_catalog_hash"],
        "publication": publication,
    }


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _publish_candidate_artifacts(
    paths_and_content: list[tuple[Path, bytes]],
    *,
    protocol_root: Path | None,
    preserved_identities: Sequence[Mapping[str, object]] = (),
) -> str:
    from Libs.utils.evaluation.operational_twin_candidate_execution import write_bytes_exclusive

    def repository_relative(path: Path) -> Path:
        candidate = path if path.is_absolute() else REPO_ROOT / path
        try:
            relative = candidate.relative_to(REPO_ROOT)
        except ValueError as error:
            raise ValueError(
                f"candidate_protocol_path_outside_repository: {candidate}"
            ) from error
        if (
            not relative.parts
            or ".." in relative.parts
            or relative.parts[0] != "Output"
            or candidate != REPO_ROOT / relative
        ):
            raise ValueError(
                f"candidate_protocol_path_not_canonical: {candidate}"
            )
        return relative

    def require_safe_path(path: Path) -> None:
        relative = repository_relative(path)
        current = REPO_ROOT
        for component in relative.parts:
            current = current / component
            if os.path.lexists(current) and current.is_symlink():
                raise ValueError(
                    "candidate_protocol_symlink_path: "
                    f"{path} contains the symlink {current}"
                )

    def ensure_directory(path: Path, created: list[Path]) -> None:
        relative = repository_relative(path)
        current = REPO_ROOT
        for component in relative.parts:
            current = current / component
            if os.path.lexists(current):
                if current.is_symlink() or not current.is_dir():
                    raise ValueError(
                        "candidate_protocol_parent_not_directory: "
                        f"{current}"
                    )
                continue
            current.mkdir()
            created.append(current)

    def require_preserved_identities() -> None:
        for identity in preserved_identities:
            relative = str(identity["path"])
            content = read_repo_bytes(REPO_ROOT, relative)
            object_id = subprocess.run(
                ["git", "rev-parse", "--verify", f"HEAD:{relative}"],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=False,
            ).stdout.strip()
            if (
                len(content) != int(identity["size_bytes"])
                or hashlib.sha256(content).hexdigest() != identity["sha256"]
                or (
                    identity.get("git_object_id") is not None
                    and object_id != identity["git_object_id"]
                )
            ):
                raise ValueError(
                    "candidate_protocol_preserved_identity_drift: "
                    f"{identity['path']} changed during publication"
                )

    require_preserved_identities()
    targets = [path for path, _ in paths_and_content]
    if len(targets) != len(set(targets)):
        raise ValueError("candidate_protocol_duplicate_publication_target")
    for path in targets:
        require_safe_path(path)
    if protocol_root is not None:
        require_safe_path(protocol_root)
    collisions = []
    if protocol_root is not None and os.path.lexists(protocol_root):
        collisions.append(protocol_root)
    collisions.extend(
        path for path, _ in paths_and_content if os.path.lexists(path)
    )
    if collisions:
        unique_collisions = list(dict.fromkeys(collisions))
        raise FileExistsError(
            "candidate_protocol_collision: candidate protocol publication is single-use; "
            "refusing every existing protocol root or output path: "
            + ", ".join(path.as_posix() for path in unique_collisions)
        )
    created: list[Path] = []
    created_directories: list[Path] = []
    created_identity: dict[Path, tuple[int, int]] = {}
    try:
        if protocol_root is not None:
            ensure_directory(protocol_root.parent, created_directories)
            protocol_root.mkdir(exist_ok=False)
            created_directories.append(protocol_root)
        for path, content in paths_and_content:
            require_safe_path(path)
            ensure_directory(path.parent, created_directories)
            write_bytes_exclusive(path, content)
            metadata = path.lstat()
            created.append(path)
            created_identity[path] = (metadata.st_dev, metadata.st_ino)
        require_preserved_identities()
        return "created"
    except BaseException:
        for path in reversed(created):
            try:
                current = path.lstat()
                if (current.st_dev, current.st_ino) == created_identity[path]:
                    path.unlink()
            except FileNotFoundError:
                pass
        for path in reversed(created_directories):
            try:
                path.rmdir()
            except OSError:
                pass
        raise


def _build_candidate_lineage(config: dict[str, object], authoritative_date: str) -> dict[str, object]:
    blockers = validate_candidate_protocol_config(config, REPO_ROOT)
    protocol = dict(config.get("operational_twin_candidate_lineage_protocol", {}))
    if protocol.get("authoritative_date") != authoritative_date:
        blockers.append(
            {
                "code": "candidate_authoritative_date_mismatch",
                "message": "CLI authoritative date must match the candidate-lineage protocol.",
            }
        )
    if blockers:
        return {
            "status": "BLOCKED",
            "terminal_decision": CANDIDATE_NEGATIVE_TERMINAL,
            "blockers": blockers,
        }
    candidate_manifest = build_candidate_lineage_manifest(config)
    assignment_manifest = build_candidate_assignment_manifest(config, candidate_manifest)
    command_catalog = build_candidate_commands(config, candidate_manifest, assignment_manifest)
    preflight = build_candidate_preflight_authority(
        config,
        REPO_ROOT,
        candidate_manifest,
        assignment_manifest,
        command_catalog,
    )
    blockers = validate_candidate_artifacts(
        config,
        REPO_ROOT,
        candidate_manifest,
        assignment_manifest,
        command_catalog,
        preflight,
        require_attempt_roots_absent=True,
    )
    if blockers:
        return {
            "status": "BLOCKED",
            "terminal_decision": CANDIDATE_NEGATIVE_TERMINAL,
            "blockers": blockers,
        }
    paths = dict(protocol["output_paths"])
    protocol_root = REPO_ROOT / str(paths["protocol_root"])
    publication = _publish_candidate_artifacts(
        [
            (REPO_ROOT / str(paths["candidate_lineage_manifest"]), _json_bytes(candidate_manifest)),
            (REPO_ROOT / str(paths["assignment_manifest"]), _json_bytes(assignment_manifest)),
            (REPO_ROOT / str(paths["assignment_commands"]), _json_bytes(command_catalog)),
            (REPO_ROOT / str(paths["preflight_authority"]), _json_bytes(preflight)),
            (REPO_ROOT / str(paths["preflight_report"]), _candidate_report(preflight).encode("utf-8")),
        ],
        protocol_root=protocol_root,
    )
    return {
        "status": "PASS",
        "terminal_decision": preflight["terminal_decision"],
        "candidate_lineage_uid": preflight["candidate_lineage_uid"],
        "protocol_uid": preflight["protocol_uid"],
        "candidate_manifest_hash": preflight["candidate_manifest_hash"],
        "assignment_manifest_hash": preflight["assignment_manifest_hash"],
        "command_catalog_hash": preflight["command_catalog_hash"],
        "assignment_count": preflight["assignment_count"],
        "publication": publication,
    }


def _build_candidate_lineage_successor_protocol(
    config: dict[str, object],
    authoritative_date: str,
    *,
    pre_change_identity_manifest_path: Path,
    validation_evidence_path: Path,
) -> dict[str, object]:
    contract = dict(config["operational_twin_successor_lineage_protocol"])
    if contract.get("authoritative_date") != authoritative_date:
        return {
            "status": "BLOCKED",
            "terminal_decision": SUCCESSOR_INVALID_TERMINAL,
            "blockers": [
                {
                    "code": "successor_authoritative_date_mismatch",
                    "message": (
                        "CLI authoritative date must match the frozen "
                        "successor-lineage protocol."
                    ),
                }
            ],
        }

    package, preserved, blockers = build_successor_package(
        config,
        REPO_ROOT,
        pre_change_manifest_path=pre_change_identity_manifest_path,
        validation_evidence_path=validation_evidence_path,
        require_publication_targets_absent=True,
        require_committed_sources=True,
    )
    if blockers:
        return {
            "status": "BLOCKED",
            "terminal_decision": SUCCESSOR_INVALID_TERMINAL,
            "blockers": blockers,
        }
    recomputed, recomputed_preserved, recomputed_blockers = build_successor_package(
        config,
        REPO_ROOT,
        pre_change_manifest_path=pre_change_identity_manifest_path,
        validation_evidence_path=validation_evidence_path,
        require_publication_targets_absent=True,
        require_committed_sources=True,
    )
    if recomputed_blockers:
        return {
            "status": "BLOCKED",
            "terminal_decision": SUCCESSOR_INVALID_TERMINAL,
            "blockers": recomputed_blockers,
        }
    if package != recomputed or preserved != recomputed_preserved:
        raise ValueError("successor_package_recomputation_mismatch")

    expected_paths = successor_publication_paths(config)
    if set(package) != set(expected_paths) or any(
        package[key][0] != expected_paths[key] for key in expected_paths
    ):
        raise ValueError("successor_publication_scope_mismatch")
    selected = [
        (REPO_ROOT / expected_paths[key], package[key][1])
        for key in expected_paths
    ]
    publication = _publish_candidate_artifacts(
        selected,
        protocol_root=REPO_ROOT
        / str(dict(contract["output_namespace"])["protocol_root"]),
        preserved_identities=preserved,
    )
    audit = json.loads(package["protocol_audit"][1].decode("utf-8"))
    return {
        "status": "PASS",
        "terminal_decision": audit["terminal_decision"],
        "candidate_lineage_uid": audit["candidate_lineage_uid"],
        "protocol_uid": audit["protocol_uid"],
        "preflight_uid": audit["preflight_uid"],
        "assignment_count": audit["assignment_count"],
        "attempt_count": audit["attempt_count"],
        "command_count": audit["command_count"],
        "publication_file_count": audit["publication_file_count"],
        "candidate_dispatch_authorized": False,
        "execution_handoff_required": True,
        "scientific_twin_result": "NOT_RUN",
        "publication": publication,
    }


def _build_candidate_retry_authority(config: dict[str, object], authoritative_date: str) -> dict[str, object]:
    contract = dict(config["operational_twin_candidate_retry_authority"])
    if contract.get("authoritative_date") != authoritative_date:
        raise ValueError("retry_authority_authoritative_date_mismatch")
    audit = adjudicate_retry_authority(config, REPO_ROOT)
    paths = dict(contract["output_paths"])
    published_audit = audit
    successor: dict[str, object] | None = None
    if audit["terminal_decision"] == RETRY_AUTHORITY_POSITIVE:
        successor = derive_positive_successor(config, REPO_ROOT, audit)
        published_audit = materialized_positive_audit(audit)
        blockers = validate_positive_successor_package(
            config,
            REPO_ROOT,
            published_audit,
            successor,
            expected_audit_materialized=True,
            require_publication_targets_absent=True,
        )
        if blockers:
            raise ValueError(
                "candidate_retry_authority_positive_prepublication_invalid: "
                + json.dumps(blockers, sort_keys=True)
            )
    selected: list[tuple[Path, bytes]] = [
        (REPO_ROOT / str(paths["decision_audit"]), _json_bytes(published_audit)),
        (REPO_ROOT / str(paths["decision_report"]), render_retry_authority_report(published_audit).encode("utf-8")),
    ]
    protocol_root: Path | None = None
    result: dict[str, object] = {
        "status": "PASS",
        "terminal_decision": published_audit["terminal_decision"],
        "next_direction": published_audit["next_direction"],
        "assignment_count": published_audit["assignment_count"],
        "authorized_row_count": published_audit["authorized_row_count"],
        "successor_required_row_count": published_audit["successor_required_row_count"],
    }
    if successor is not None:
        protocol_root = REPO_ROOT / str(paths["protocol_root"])
        selected.extend([
            (REPO_ROOT / str(paths["candidate_lineage_manifest"]), _json_bytes(successor["candidate_lineage_manifest"])),
            (REPO_ROOT / str(paths["assignment_manifest"]), _json_bytes(successor["assignment_manifest"])),
            (REPO_ROOT / str(paths["assignment_commands"]), _json_bytes(successor["assignment_commands"])),
            (REPO_ROOT / str(paths["preflight_authority"]), _json_bytes(successor["preflight_authority"])),
            (REPO_ROOT / str(paths["preflight_report"]), render_positive_successor_report(successor["preflight_authority"]).encode("utf-8")),
        ])
        result.update(
            protocol_uid=successor["candidate_lineage_manifest"]["protocol_uid"],
            preflight_uid=successor["candidate_lineage_manifest"]["preflight_uid"],
            candidate_manifest_hash=successor["candidate_lineage_manifest"]["candidate_manifest_hash"],
            assignment_manifest_hash=successor["assignment_manifest"]["assignment_manifest_hash"],
            command_catalog_hash=successor["assignment_commands"]["command_catalog_hash"],
        )
    result["publication"] = _publish_candidate_artifacts(selected, protocol_root=protocol_root)
    return result


def _build_candidate_successor_execution_handoff(
    config: Mapping[str, object], authoritative_date: str,
    *, pre_change_identity_manifest_path: Path, validation_evidence_path: Path,
) -> dict[str, object]:
    from Libs.utils.evaluation.operational_twin_successor_execution_handoff import (
        CONFIG_KEY, _path, build_handoff_publications,
    )
    contract = config[CONFIG_KEY]
    if authoritative_date != contract["authoritative_date"]:
        raise ValueError("handoff_authoritative_date_mismatch")
    for supplied, name in ((pre_change_identity_manifest_path, "pre_change_identity_manifest"),
                           (validation_evidence_path, "source_gate")):
        if supplied.absolute() != _path(REPO_ROOT, contract["evidence_inputs"][name]):
            raise ValueError("handoff_builder_evidence_path_mismatch")
    source_gate = json.loads(validation_evidence_path.read_text())
    source_commit = source_gate["source_identity"]["source_commit"]
    publications = build_handoff_publications(config, REPO_ROOT, source_commit=source_commit)
    selected = [(REPO_ROOT / relative, content) for relative, content in publications.values()]
    manifest = json.loads(publications["manifest"][1])
    publication = _publish_candidate_artifacts(
        selected, protocol_root=(REPO_ROOT / contract["output_paths"]["manifest"]).parent,
        preserved_identities=list(manifest["frozen_package"]["artifact_identities"].values()),
    )
    return {"status": "PASS", "terminal_decision": manifest["terminal_decision"],
            "handoff_uid": manifest["handoff_uid"], "executing_source_commit": source_commit,
            "candidate_dispatch_authorized": False, "scientific_twin_result": "NOT_RUN",
            "publication": publication}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "Libs/configs/config_stack.yaml")
    parser.add_argument(
        "--protocol-scope",
        choices=(
            "historical-v1",
            "manuscript-native",
            "candidate-lineage",
            "candidate-lineage-resource-receipt-amendment",
            "candidate-lineage-execution-interface-repair",
            "candidate-lineage-retry-authority",
            "candidate-lineage-successor-protocol",
            "candidate-lineage-successor-execution-handoff",
        ),
        default="historical-v1",
    )
    parser.add_argument("--authoritative-date", required=True)
    parser.add_argument("--output-dir", type=Path, help="Manuscript-native non-dispatch package destination")
    parser.add_argument("--sinfo-path", type=Path)
    parser.add_argument("--sacct-path", type=Path)
    parser.add_argument("--cpu-probe-output-path", type=Path)
    parser.add_argument("--gpu-probe-output-path", type=Path)
    parser.add_argument("--cpu-probe-job-id")
    parser.add_argument("--gpu-probe-job-id")
    parser.add_argument("--cpu-live-probe-receipt", type=Path)
    parser.add_argument("--gpu-live-probe-receipt", type=Path)
    parser.add_argument("--array-resource-fixture", type=Path)
    parser.add_argument("--array-terminal-fixture", type=Path)
    parser.add_argument("--pre-change-identity-manifest", type=Path)
    parser.add_argument("--validation-evidence", type=Path)
    args = parser.parse_args()

    config = load_config_stack(args.config.resolve())
    if args.protocol_scope == "manuscript-native":
        try:
            result = publish_package(config, REPO_ROOT, args.authoritative_date, output_dir=args.output_dir)
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            result = {"status":"BLOCKED","error":str(error),"scientific_twin":"NOT_RUN",
                      "formal_qualification":False,"dispatch_authorized":False}
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "PASS" else 2

    if args.protocol_scope == "candidate-lineage-successor-execution-handoff":
        if args.pre_change_identity_manifest is None or args.validation_evidence is None:
            parser.error("successor execution handoff requires --pre-change-identity-manifest and --validation-evidence")
        try:
            result = _build_candidate_successor_execution_handoff(
                config, args.authoritative_date,
                pre_change_identity_manifest_path=args.pre_change_identity_manifest,
                validation_evidence_path=args.validation_evidence,
            )
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            result = {
                "status": "BLOCKED",
                "terminal_decision": "OPERATIONAL_TWIN_SUCCESSOR_EXECUTION_HANDOFF_BLOCKED",
                "candidate_dispatch_authorized": False, "scientific_twin_result": "NOT_RUN",
                "blockers": [{"code": "handoff_publication_failed", "message": str(error)}],
            }
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "PASS" else 2
    if args.protocol_scope == "candidate-lineage-successor-protocol":
        required_successor_args = {
            "pre_change_identity_manifest": args.pre_change_identity_manifest,
            "validation_evidence": args.validation_evidence,
        }
        missing = [
            name for name, value in required_successor_args.items() if value is None
        ]
        if missing:
            parser.error(
                "candidate-lineage-successor-protocol requires: "
                + ", ".join(
                    f"--{name.replace('_', '-')}" for name in missing
                )
            )
        try:
            result = _build_candidate_lineage_successor_protocol(
                config,
                args.authoritative_date,
                pre_change_identity_manifest_path=(
                    args.pre_change_identity_manifest.resolve()
                ),
                validation_evidence_path=args.validation_evidence.resolve(),
            )
        except (FileExistsError, OSError, ValueError, RuntimeError) as error:
            result = {
                "status": "BLOCKED",
                "terminal_decision": SUCCESSOR_INVALID_TERMINAL,
                "blockers": [
                    {
                        "code": "successor_protocol_publication_failed",
                        "message": str(error),
                    }
                ],
            }
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "PASS" else 2
    if args.protocol_scope == "candidate-lineage-retry-authority":
        try:
            result = _build_candidate_retry_authority(config, args.authoritative_date)
        except (FileExistsError, OSError, ValueError, RuntimeError) as error:
            result = {
                "status": "BLOCKED",
                "terminal_decision": None,
                "blockers": [
                    {
                        "code": "candidate_retry_authority_publication_failed",
                        "message": str(error),
                    }
                ],
            }
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "PASS" else 2
    if args.protocol_scope == "candidate-lineage-execution-interface-repair":
        required_repair_args = {
            "array_resource_fixture": args.array_resource_fixture,
            "array_terminal_fixture": args.array_terminal_fixture,
            "pre_change_identity_manifest": args.pre_change_identity_manifest,
            "validation_evidence": args.validation_evidence,
        }
        missing = [
            name for name, value in required_repair_args.items() if value is None
        ]
        if missing:
            parser.error(
                "candidate-lineage-execution-interface-repair requires: "
                + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            )
        repair = dict(config["operational_twin_candidate_execution"])[
            "interface_repair_contract"
        ]
        try:
            result = _build_candidate_execution_interface_repair(
                config,
                args.authoritative_date,
                array_resource_fixture_path=args.array_resource_fixture.resolve(),
                array_terminal_fixture_path=args.array_terminal_fixture.resolve(),
                pre_change_identity_manifest_path=(
                    args.pre_change_identity_manifest.resolve()
                ),
                validation_evidence_path=args.validation_evidence.resolve(),
            )
        except (FileExistsError, OSError, ValueError) as error:
            result = {
                "status": "BLOCKED",
                "terminal_decision": repair["negative_terminal"],
                "blockers": [
                    {
                        "code": "candidate_protocol_publication_failed",
                        "message": str(error),
                    }
                ],
            }
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "PASS" else 2
    current_candidate_schema = str(
        dict(config["operational_twin_candidate_lineage_protocol"])[
            "schema_version"
        ]
    )
    if (
        current_candidate_schema
        == "neuronsifter-operational-twin-candidate-lineage-protocol-v3"
        and args.protocol_scope
        in {
            "candidate-lineage",
            "candidate-lineage-resource-receipt-amendment",
        }
    ):
        repair = dict(config["operational_twin_candidate_execution"])[
            "interface_repair_contract"
        ]
        print(
            json.dumps(
                {
                    "status": "BLOCKED",
                    "terminal_decision": repair["negative_terminal"],
                    "blockers": [
                        {
                            "code": "candidate_protocol_scope_superseded",
                            "message": (
                                "Protocol v3 may be published only through "
                                "candidate-lineage-execution-interface-repair."
                            ),
                        }
                    ],
                },
                sort_keys=True,
            )
        )
        return 2
    if args.protocol_scope == "candidate-lineage-resource-receipt-amendment":
        required_amendment_args = {
            "cpu_live_probe_receipt": args.cpu_live_probe_receipt,
            "gpu_live_probe_receipt": args.gpu_live_probe_receipt,
            "validation_evidence": args.validation_evidence,
        }
        missing = [
            name for name, value in required_amendment_args.items() if value is None
        ]
        if missing:
            parser.error(
                "candidate-lineage-resource-receipt-amendment requires: "
                + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            )
        try:
            result = _build_candidate_receipt_amendment(
                config,
                args.authoritative_date,
                cpu_probe_receipt_path=args.cpu_live_probe_receipt.resolve(),
                gpu_probe_receipt_path=args.gpu_live_probe_receipt.resolve(),
                validation_evidence_path=args.validation_evidence.resolve(),
            )
        except (FileExistsError, OSError, ValueError) as error:
            result = {
                "status": "BLOCKED",
                "terminal_decision": "OPERATIONAL_TWIN_RESOURCE_RECEIPT_SEMANTICS_INVALID",
                "blockers": [
                    {
                        "code": "candidate_protocol_publication_failed",
                        "message": str(error),
                    }
                ],
            }
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "PASS" else 2
    if args.protocol_scope == "candidate-lineage":
        try:
            result = _build_candidate_lineage(config, args.authoritative_date)
        except (FileExistsError, OSError, ValueError) as error:
            result = {
                "status": "BLOCKED",
                "terminal_decision": CANDIDATE_NEGATIVE_TERMINAL,
                "blockers": [{"code": "candidate_protocol_publication_failed", "message": str(error)}],
            }
        print(json.dumps(result, sort_keys=True))
        return 0 if result["status"] == "PASS" else 2
    required_legacy_args = {
        "sinfo_path": args.sinfo_path,
        "sacct_path": args.sacct_path,
        "cpu_probe_output_path": args.cpu_probe_output_path,
        "gpu_probe_output_path": args.gpu_probe_output_path,
        "cpu_probe_job_id": args.cpu_probe_job_id,
        "gpu_probe_job_id": args.gpu_probe_job_id,
    }
    missing = [name for name, value in required_legacy_args.items() if value is None]
    if missing:
        parser.error("historical-v1 scope requires: " + ", ".join(f"--{name.replace('_', '-')}" for name in missing))
    protocol = dict(config["operational_twin_training_protocol"])
    blockers = validate_protocol_config(config, REPO_ROOT)
    open_blockers, open_authority = validate_open_data_binding(config, REPO_ROOT)
    blockers.extend(open_blockers)
    if str(protocol.get("authoritative_date")) != args.authoritative_date:
        blockers.append({"code": "authoritative_date_mismatch", "message": "CLI authoritative date must match the frozen protocol."})
    if blockers:
        print(json.dumps({"status": "BLOCKED", "blockers": blockers}, sort_keys=True))
        return 2

    sinfo_text = args.sinfo_path.read_text(encoding="utf-8")
    sacct_text = args.sacct_path.read_text(encoding="utf-8")
    cpu_probe_output = args.cpu_probe_output_path.read_text(encoding="utf-8")
    gpu_probe_output = args.gpu_probe_output_path.read_text(encoding="utf-8")
    cpu_probe_sha = sha256_file(args.cpu_probe_output_path)
    gpu_probe_sha = sha256_file(args.gpu_probe_output_path)
    if f"SLURM_JOB_ID={args.gpu_probe_job_id}" not in gpu_probe_output:
        print(json.dumps({"status": "BLOCKED", "blockers": [{"code": "gpu_probe_stdout_job_id_mismatch", "message": "GPU probe stdout does not contain the supplied Slurm job id."}]}, sort_keys=True))
        return 2
    gpu_sacct_lines = "\n".join(line for line in sacct_text.splitlines() if args.gpu_probe_job_id in line)
    if "COMPLETED" not in gpu_sacct_lines or "|0:0|" not in gpu_sacct_lines:
        print(json.dumps({"status": "BLOCKED", "blockers": [{"code": "gpu_probe_sacct_not_terminal", "message": "GPU probe sacct evidence is not terminal successful."}]}, sort_keys=True))
        return 2
    probes = {
        "cpu_probe": {
            "job_id": args.cpu_probe_job_id,
            "job_name": "NrnSift_protocol_cpu_probe",
            "partition": "cpu",
            "node": "cpu04",
            "gpu": None,
            "cpus_per_task": 1,
            "memory": "2G",
            "command": (
                "srun --partition=cpu --nodelist=cpu04 --job-name=NrnSift_protocol_cpu_probe "
                "--cpus-per-task=1 --mem=2G --time=00:02:00 /bin/sh -lc '<print environment>' "
                f"> {args.cpu_probe_output_path.as_posix()}"
            ),
            "exit_status": 0,
            "stdout_path": args.cpu_probe_output_path.as_posix(),
            "stdout_sha256": cpu_probe_sha,
            "environment": {"builder_python": sys.version.split()[0], "probe_stdout": cpu_probe_output},
            "sacct_evidence": "\n".join(line for line in sacct_text.splitlines() if args.cpu_probe_job_id in line),
        },
        "gpu_probe": {
            "job_id": args.gpu_probe_job_id,
            "job_name": "NrnSift_protocol_gpu_probe_retry",
            "partition": "gpu1",
            "node": "gpu03",
            "gpu": "NVIDIA L40S",
            "cpus_per_task": 1,
            "memory": "4G",
            "command": (
                "srun --partition=gpu1 --nodelist=gpu03 --job-name=NrnSift_protocol_gpu_probe_retry "
                "--cpus-per-task=1 --mem=4G --gres=gpu:1 --time=00:03:00 /bin/sh -lc '<print slurm/cuda/torch environment>' "
                f"> {args.gpu_probe_output_path.as_posix()}"
            ),
            "exit_status": 0,
            "stdout_path": args.gpu_probe_output_path.as_posix(),
            "stdout_sha256": gpu_probe_sha,
            "environment": {"builder_python": sys.version.split()[0], "probe_stdout": gpu_probe_output},
            "sacct_evidence": gpu_sacct_lines,
        },
    }
    assignment_manifest = build_assignment_manifest(config, REPO_ROOT)
    resource_receipt = build_resource_receipt(config, REPO_ROOT, sinfo_text=sinfo_text, sacct_text=sacct_text, probe_outputs=probes)
    preflight = build_preflight_authority(config, REPO_ROOT, assignment_manifest, resource_receipt, open_authority)
    output_roots = dict(protocol["output_roots"])
    _write_json(REPO_ROOT / output_roots["assignment_manifest"], assignment_manifest)
    _write_json(REPO_ROOT / output_roots["resource_receipt"], resource_receipt)
    _write_json(REPO_ROOT / output_roots["preflight_authority"], preflight)
    _write_report(REPO_ROOT / output_roots["preflight_report"], preflight, resource_receipt)
    print(json.dumps({"status": "PASS", "assignment_count": assignment_manifest["assignment_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
