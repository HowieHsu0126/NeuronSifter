"""Freeze and validate the Operational Twin training protocol authority."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from copy import deepcopy
import errno
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import socket
import stat
import subprocess
from typing import Mapping

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
import yaml

from Libs.utils.data.controlled_world_open_package import reject_forbidden_path_tokens


REQUIRED_METHOD_IDS = [
    "persistence_no_action",
    "linear_or_delta",
    "same_backbone_no_action",
    "deterministic_action_conditioned",
    "full_uncertainty_aware_action_conditioned",
]
METHOD_EXECUTION_CONTRACTS = {
    "persistence_no_action": {
        "execution_contract": "no_fit_persistence",
        "runtime_family": "nontrainable_baseline",
        "action_conditioning": "no_action_only",
        "metrics_path": "metrics/persistence_no_action_metrics.jsonl",
        "failure_retention_path": "failures/persistence_no_action_failures.jsonl",
        "checkpoint_creation_allowed": False,
        "loss_name": "masked_mse_no_fit",
        "calibration_eligible": False,
    },
    "linear_or_delta": {
        "execution_contract": "trainable_linear_delta",
        "runtime_family": "linear_delta_baseline",
        "action_conditioning": "structured_action_delta_features",
        "metrics_path": "metrics/linear_or_delta_metrics.jsonl",
        "failure_retention_path": "failures/linear_or_delta_failures.jsonl",
        "checkpoint_creation_allowed": True,
        "loss_name": "masked_mse_linear_delta",
        "calibration_eligible": False,
    },
    "same_backbone_no_action": {
        "execution_contract": "trainable_operational_twin_backbone_no_action",
        "runtime_family": "operational_twin_backbone_no_action_ablation",
        "action_conditioning": "no_action_ablation",
        "metrics_path": "metrics/same_backbone_no_action_metrics.jsonl",
        "failure_retention_path": "failures/same_backbone_no_action_failures.jsonl",
        "checkpoint_creation_allowed": True,
        "loss_name": "masked_mse_no_action_backbone",
        "calibration_eligible": False,
    },
    "deterministic_action_conditioned": {
        "execution_contract": "trainable_action_conditioned_mean",
        "runtime_family": "operational_twin_deterministic_action_conditioned",
        "action_conditioning": "structured_world_action_mean_only",
        "metrics_path": "metrics/deterministic_action_conditioned_metrics.jsonl",
        "failure_retention_path": "failures/deterministic_action_conditioned_failures.jsonl",
        "checkpoint_creation_allowed": True,
        "loss_name": "masked_mse_deterministic",
        "calibration_eligible": False,
    },
    "full_uncertainty_aware_action_conditioned": {
        "execution_contract": "trainable_action_conditioned_distributional",
        "runtime_family": "operational_twin_uncertainty_aware_action_conditioned",
        "action_conditioning": "structured_world_action_distributional",
        "metrics_path": "metrics/full_uncertainty_aware_action_conditioned_metrics.jsonl",
        "failure_retention_path": "failures/full_uncertainty_aware_action_conditioned_failures.jsonl",
        "checkpoint_creation_allowed": True,
        "loss_name": "masked_gaussian_nll",
        "calibration_eligible": True,
    },
}
REQUIRED_ZERO_ACCESS_COUNTERS = [
    "sealed_material_access",
    "test_outcome_access",
    "protected_holdout_access",
    "six_context_outcome_access",
    "tool_access",
    "mcp_access",
    "agent_trace_access",
    "agent_access",
    "memory_access",
    "coevolution_access",
]
FORBIDDEN_PATH_TOKENS = ("sealed", "protected", "six_context", "test")
QUALIFIED_TERMINAL = "OPEN_TWIN_DEVELOPMENT_DATA_QUALIFIED"
CANDIDATE_PROTOCOL_KEY = "operational_twin_candidate_lineage_protocol"
CANDIDATE_SCHEMA_KEY = "operational_twin_candidate_lineage_protocol_schema"
CANDIDATE_POSITIVE_TERMINAL = (
    "OPERATIONAL_TWIN_CANDIDATE_EXECUTION_INTERFACES_REPAIRED"
)
CANDIDATE_NEGATIVE_TERMINAL = (
    "OPERATIONAL_TWIN_CANDIDATE_EXECUTION_INTERFACES_NOT_REPAIRED"
)
CANDIDATE_EXECUTION_AUTHORITY_TERMINAL = (
    "OPERATIONAL_TWIN_CANDIDATE_EXECUTION_AUTHORIZED"
)
CANDIDATE_RESOURCE_RECEIPT_TERMINAL = (
    "OPERATIONAL_TWIN_CANDIDATE_RESOURCE_ENVIRONMENT_RECEIPT_VALID"
)
CANDIDATE_PROTOCOL_SCHEMA_VERSION = (
    "neuronsifter-operational-twin-candidate-lineage-protocol-v3"
)
CANDIDATE_MANIFEST_SCHEMA_VERSION = (
    "neuronsifter-operational-twin-candidate-lineage-manifest-v3"
)
CANDIDATE_PREFLIGHT_SCHEMA_VERSION = (
    "neuronsifter-operational-twin-candidate-lineage-preflight-v3"
)
CANDIDATE_RESOURCE_RECEIPT_SCHEMA_VERSION_V1 = (
    "neuronsifter-operational-twin-candidate-resource-environment-receipt-v1"
)
CANDIDATE_RESOURCE_RECEIPT_SCHEMA_VERSION_V2 = (
    "neuronsifter-operational-twin-candidate-resource-environment-receipt-v2"
)
CANDIDATE_RESOURCE_RECEIPT_SCHEMA_VERSION_V3 = (
    "neuronsifter-operational-twin-candidate-resource-environment-receipt-v3"
)
CANDIDATE_EXECUTION_AUTHORITY_SCHEMA_VERSION_V1 = (
    "neuronsifter-operational-twin-candidate-execution-authority-v1"
)
CANDIDATE_EXECUTION_AUTHORITY_SCHEMA_VERSION_V2 = (
    "neuronsifter-operational-twin-candidate-execution-authority-v2"
)
CANDIDATE_EXECUTION_AUTHORITY_SCHEMA_VERSION_V3 = (
    "neuronsifter-operational-twin-candidate-execution-authority-v3"
)
CANDIDATE_SCHEDULER_EVIDENCE_CONTRACT_VERSION = (
    "slurm-22.05.9-execution-interface-v3"
)
CANDIDATE_SCHEDULER_FIELD_MAPPING = {
    "resource_query_job_id": "derived(SLURM_ARRAY_JOB_ID,SLURM_ARRAY_TASK_ID,SLURM_JOB_ID)",
    "scontrol_job_id": "JobId",
    "scontrol_array_job_id": "ArrayJobId",
    "scontrol_array_task_id": "ArrayTaskId",
    "partition": "Partition",
    "nodelist": "NodeList",
    "cpus_per_task": "CPUs/Task",
    "tres_cpu": "TRES.cpu",
    "memory": "MinMemoryNode",
    "oversubscribe": "OverSubscribe",
    "gpu_gres_per_node": "TresPerNode",
}
CANDIDATE_FORBIDDEN_LEGACY_SCHEDULER_FIELDS = [
    "ReqTRES",
    "AllocTRES",
    "Shared",
]
CANDIDATE_IDENTITY_FIELDS = (
    "candidate_lineage_uid",
    "protocol_uid",
    "assignment_uid",
    "attempt_uid",
)
CANDIDATE_FORBIDDEN_STATE_INPUT_FIELDS = (
    "resume_input",
    "history_input",
    "checkpoint_input",
    "optimizer_state_input",
    "model_state_input",
    "calibration_state_input",
)
CANDIDATE_SCIENTIFIC_EQUALITY_FIELDS = (
    "required_open_data_terminal",
    "open_data_authority",
    "open_data_authority_training_authorized",
    "manifests",
    "methods",
    "seeds",
    "seed_policy",
    "runtime",
    "selection_and_calibration",
    "statistics",
    "failure_policy",
)
CANDIDATE_EXECUTION_AUTHORITY_KEYS = {
    "schema_version",
    "status",
    "terminal_decision",
    "authority_uid",
    "issued_at",
    "authority_role",
    "approved_protocol_review_identity",
    "approved_protocol_review_decision",
    "candidate_lineage_uid",
    "protocol_uid",
    "preflight_uid",
    "candidate_manifest_hash",
    "assignment_manifest_hash",
    "command_catalog_hash",
    "source_commit",
    "source_identity_hash",
    "assignment_uid",
    "attempt_uid",
    "method_id",
    "seed",
    "output_root",
    "resource_profile",
    "resource_requirements",
    "resource_environment_receipt_identity",
    "fresh_initialization_receipt",
    "state_resume_allowed",
    "training_execution_authorized",
    "checkpoint_creation_authorized",
    "calibration_fitting_authorized",
    "access_counters",
}
CANDIDATE_RESOURCE_RECEIPT_KEYS = {
    "schema_version",
    "status",
    "terminal_decision",
    "receipt_uid",
    "created_at",
    "sinfo_captured_at",
    "sinfo_command",
    "sinfo_stdout",
    "sinfo_stdout_sha256",
    "resource_captured_at",
    "resource_command",
    "resource_stdout",
    "resource_stdout_sha256",
    "resource_allocation",
    "candidate_lineage_uid",
    "protocol_uid",
    "preflight_uid",
    "candidate_manifest_hash",
    "assignment_manifest_hash",
    "command_catalog_hash",
    "source_commit",
    "source_identity_hash",
    "assignment_uid",
    "attempt_uid",
    "method_id",
    "seed",
    "output_root",
    "resource_profile",
    "resource_requirements",
    "slurm",
    "environment",
    "fresh_sinfo",
    "training_executed",
    "calibration_fitting_executed",
    "checkpoint_created",
    "access_counters",
}
CANDIDATE_RESOURCE_SLURM_KEYS = {
    "slurm_job_id",
    "slurm_array_job_id",
    "slurm_array_task_id",
    "slurm_job_name",
    "slurm_job_partition",
    "slurm_job_nodelist",
    "slurm_cpus_per_task",
    "slurm_mem_per_node",
    "slurm_mem_per_cpu",
    "slurm_job_gres",
    "hostname",
    "slurm_job_gpus",
    "slurm_gpus_on_node",
    "slurm_step_gpus",
    "resource_query_job_id",
}
CANDIDATE_RESOURCE_ALLOCATION_KEYS = {
    "partition",
    "nodelist",
    "cpus_per_task",
    "memory",
    "gres",
    "exclusive",
}
CANDIDATE_RESOURCE_ENVIRONMENT_KEYS = {
    "hostname",
    "python_version",
    "torch_version",
    "cuda_available",
    "cuda_device_count",
    "cuda_visible_devices",
}
CANDIDATE_COMMAND_ROW_KEYS = {
    "command_index",
    "candidate_lineage_uid",
    "protocol_uid",
    "assignment_uid",
    "attempt_uid",
    "attempt_kind",
    "conditional",
    "eligible_after_failure_class",
    "predecessor_attempt_uid",
    "method_id",
    "seed",
    "resource_profile",
    "resource_requirements",
    "wall_time",
    "output_root",
    "execution_authority",
    "resource_environment_receipt",
    "argv",
}
CANDIDATE_EXECUTION_AUTHORITY_BINDING_KEYS = {
    "path",
    "sha256",
    "git_object_id",
    "authority_uid",
    "source_commit",
    "approved_protocol_review_identity",
}
CANDIDATE_RESOURCE_RECEIPT_BINDING_KEYS = {
    "path",
    "sha256",
    "git_object_id",
    "receipt_uid",
    "source_commit",
    "scheduler",
    "environment",
    "resource_allocation",
}
CANDIDATE_RESOURCE_PROFILES = {
    "cpu_light": {
        "partition": "cpu",
        "nodelist": "cpu04",
        "cpus_per_task": 2,
        "memory": "8G",
        "gres": None,
        "exclusive": False,
    },
    "gpu_single": {
        "partition": "gpu1",
        "nodelist": "gpu03",
        "cpus_per_task": 6,
        "memory": "32G",
        "gres": "gpu:1",
        "exclusive": False,
    },
}
CANDIDATE_SUPERSESSION_CONTRACT = {
    "predecessor_implementation_commit": "38ec52f9800b67409b768fe86f93d9167f105838",
    "predecessor_candidate_lineage_uid": "operational_twin_candidate_lineage_ef80556d38c9e1b50c28353a",
    "predecessor_protocol_uid": "operational_twin_candidate_protocol_43a48868a277888e0628d671",
    "rejected_review_record": {
        "path": "orchestrator/rounds/round-43-freeze-operational-twin-candidate-lineage-protocol/review-record.json",
        "sha256": "9ba42915171d79c4de487ebd4731b3f65d0a793f7018b99f457fc0e68b497a93",
        "git_object_id": "d997d19b42030a5db1bcdc3534ee3dce720415f9",
        "schema_version": "review-record-v4",
        "round_id": "round-43-freeze-operational-twin-candidate-lineage-protocol",
        "direction_id": "freeze-operational-twin-candidate-lineage-protocol",
        "decision": "rejected",
    },
    "predecessor_artifacts": {
        "candidate_lineage_manifest": {
            "path": "Output/results/operational_twin/candidate_lineage_protocol/execution_guarded/candidate_lineage_manifest.json",
            "sha256": "ccc75eb9ee8070918dc10dae9b94a62d7cfb94fba74b8f72cc502284aa374d6e",
        },
        "assignment_manifest": {
            "path": "Output/results/operational_twin/candidate_lineage_protocol/execution_guarded/assignment_manifest.json",
            "sha256": "fbfcf52ec6d4573c52cb567c43524b6c9a9e88d22defaf1252f0d0db74a4fa8c",
        },
        "assignment_commands": {
            "path": "Output/results/operational_twin/candidate_lineage_protocol/execution_guarded/assignment_commands.json",
            "sha256": "f2f8c119ca0c63720d169a4a30e9299a48d3996f1455019219f9478d718a187c",
        },
        "preflight_authority": {
            "path": "Output/audits/operational_twin_candidate_lineage_execution_guarded_preflight.json",
            "sha256": "402ba33693d9a348368d13f69b052a2875c947e28f44749baeb94b88077a6e31",
        },
        "preflight_report": {
            "path": "Output/reports/operational_twin_candidate_lineage_execution_guarded_preflight.md",
            "sha256": "192b156969913178664494a124370cd84f5c01d81585101f7ed110499e5cb08d",
        },
    },
    "preservation_policy": {
        "predecessor_paths_preserved": True,
        "predecessor_bytes_preserved": True,
        "successor_paths_disjoint": True,
        "predecessor_overwrite_allowed": False,
    },
}
CANDIDATE_SUPERSESSION_CONTRACT["resource_receipt_amendment_basis"] = {
    "merged_round_44_commit": "af3a677194bd280d641c7ecaef618ffecb538b3d",
    "approved_round_43_review_record": {
        "path": "orchestrator/rounds/round-43-freeze-operational-twin-candidate-lineage-protocol/review-record.json",
        "sha256": "96d1aaa3d605195a234331d296548f7b36e8f0ebd4acee4192003abc09ac83cb",
        "git_object_id": "442e1f69c8565f6e79276591644274f52915fa09",
        "schema_version": "review-record-v4",
        "round_id": "round-43-freeze-operational-twin-candidate-lineage-protocol",
        "direction_id": "freeze-operational-twin-candidate-lineage-protocol",
        "decision": "approved",
    },
    "approved_round_44_review_record": {
        "path": "orchestrator/rounds/round-44-execute-operational-twin-candidate-lineage/review-record.json",
        "sha256": "d7f0cb15f07d30a97f7709c870e8328f7c6166b5619e0ebaddc3e5e00d452693",
        "git_object_id": "a597aea5edfe389436a7927b5e2fda52768df1b9",
        "schema_version": "review-record-v4",
        "round_id": "round-44-execute-operational-twin-candidate-lineage",
        "direction_id": "execute-operational-twin-candidate-lineage",
        "decision": "approved",
    },
    "approved_rev_007_update": {
        "path": "orchestrator/roadmap-updates/round-44-execute-operational-twin-candidate-lineage-roadmap-update.md",
        "sha256": "b60e48ba152b7393bc25f63a06eef690c7c1e6978d5580e5fdb0be399ed7eaa5",
        "git_object_id": "d90decc3b4e390defc7029b82d23a7f5ca06b87e",
    },
    "approved_rev_007_update_review": {
        "path": "orchestrator/roadmap-updates/round-44-execute-operational-twin-candidate-lineage-roadmap-update-review.md",
        "sha256": "9a2c44d6857c90e76fce8b32b5e72f87114b6451a8c96faad39caf0974a249f5",
        "git_object_id": "8e4fc1f2d860c9d352098a7a1e3d58dad428b5b4",
        "decision": "APPROVED",
    },
    "predecessor_generations": {
        "original": {
            "protocol_uid": "operational_twin_candidate_protocol_77566c1a2dcc1771cfbeb830",
            "artifacts": {
                "candidate_lineage_manifest": {
                    "path": "Output/results/operational_twin/candidate_lineage_protocol/candidate_lineage_manifest.json",
                    "sha256": "8ceff262ddd01eb2d85ac52d0cd3454912de91b0bafdd9da420f4a499768c167",
                },
                "assignment_manifest": {
                    "path": "Output/results/operational_twin/candidate_lineage_protocol/assignment_manifest.json",
                    "sha256": "9829484042200d0bc75ed70daaa858323eb96f71a641060511661887a951ae3b",
                },
                "assignment_commands": {
                    "path": "Output/results/operational_twin/candidate_lineage_protocol/assignment_commands.json",
                    "sha256": "91a4eda57e61bdeae1807acd4902ae903cc2609e76b00e65fcc715db5204e9e7",
                },
                "preflight_authority": {
                    "path": "Output/audits/operational_twin_candidate_lineage_protocol_preflight.json",
                    "sha256": "fc63c83a1562efe7a0f8921fada866f351bbcc7381c24d8aeba56415f2113553",
                },
                "preflight_report": {
                    "path": "Output/reports/operational_twin_candidate_lineage_protocol_preflight.md",
                    "sha256": "ec52ae509a020a4da113bd3f9349ff2bba5a3ac41c129fc3664344bbe8812e24",
                },
            },
        },
        "execution_guarded": {
            "protocol_uid": "operational_twin_candidate_protocol_43a48868a277888e0628d671",
            "artifacts": deepcopy(
                CANDIDATE_SUPERSESSION_CONTRACT["predecessor_artifacts"]
            ),
        },
        "dispatch_guarded": {
            "protocol_uid": "operational_twin_candidate_protocol_c7e36201c1534c954f888337",
            "artifacts": {
                "candidate_lineage_manifest": {
                    "path": "Output/results/operational_twin/candidate_lineage_protocol/dispatch_guarded/candidate_lineage_manifest.json",
                    "sha256": "ed803a56fcb9570c6a841e77a5ec5f9e226473b1c489a017579d3948627db8a6",
                },
                "assignment_manifest": {
                    "path": "Output/results/operational_twin/candidate_lineage_protocol/dispatch_guarded/assignment_manifest.json",
                    "sha256": "b73442a467bb8f1b95cf8dd22e9f244147d05d9e6a7c182c4c8afaa5ebb3b12c",
                },
                "assignment_commands": {
                    "path": "Output/results/operational_twin/candidate_lineage_protocol/dispatch_guarded/assignment_commands.json",
                    "sha256": "fa6b1be03afb471966cbe980d7ae38c36f7c7cfcba5a9076db2c789e3653fc4c",
                },
                "preflight_authority": {
                    "path": "Output/audits/operational_twin_candidate_lineage_dispatch_guarded_preflight.json",
                    "sha256": "e56ac78d57849b0ee5a09b2ec6226381c4ea7e52dfc940dcbef388861c41bbcd",
                },
                "preflight_report": {
                    "path": "Output/reports/operational_twin_candidate_lineage_dispatch_guarded_preflight.md",
                    "sha256": "c3631eaaf55dba02699caf7b167e543d4bd564ae2474886352085457d28a643c",
                },
            },
        },
    },
}
CANDIDATE_SUPERSESSION_CONTRACT["resource_receipt_amendment_basis"][
    "predecessor_generations"
]["resource_receipt_guarded"] = {
    "protocol_uid": "operational_twin_candidate_protocol_2dd00f48bee88f20cacb4882",
    "artifacts": {
        "candidate_lineage_manifest": {
            "path": "Output/results/operational_twin/candidate_lineage_protocol/resource_receipt_guarded/candidate_lineage_manifest.json",
            "sha256": "987801b967a016337c483e57323ae915187550d6aa67914461a081f63b3c0536",
        },
        "assignment_manifest": {
            "path": "Output/results/operational_twin/candidate_lineage_protocol/resource_receipt_guarded/assignment_manifest.json",
            "sha256": "5f3310f4d273451a7b055f147cd48027c2a9dd611f1305264ad17934ad383977",
        },
        "assignment_commands": {
            "path": "Output/results/operational_twin/candidate_lineage_protocol/resource_receipt_guarded/assignment_commands.json",
            "sha256": "57a221893b576fc10926da9176cb877d4a00d0fb6a838599638be8696469c9be",
        },
        "preflight_authority": {
            "path": "Output/audits/operational_twin_candidate_lineage_resource_receipt_guarded_preflight.json",
            "sha256": "13298f296543dfbcb44182529b90c00b8ffe85aab2b6710b64c965d84846df6b",
        },
        "preflight_report": {
            "path": "Output/reports/operational_twin_candidate_lineage_resource_receipt_guarded_preflight.md",
            "sha256": "069a2bebbfe21aa71379133ff96177a23cd6e4bddc8b8c7495baa473c75ac781",
        },
    },
}
CANDIDATE_SUPERSESSION_CONTRACT["execution_interface_repair_basis"] = {
    "baseline_commit": "3ae8875a547786beb65604fe5cc4bc09d4103872",
    "merged_round_46_commit": "d2d57c9f5a08b5933b66e395f6128de68aacf16f",
    "round_46_execution_commit": "a3c4bb054617d9c7a84e19d86003eeddd668bdb0",
    "approved_round_46_review_record": {
        "path": "orchestrator/rounds/round-46-execute-operational-twin-candidate-lineage/review-record.json",
        "sha256": "03d83b523257beebf35c73d52a1e6136b4d70da9dd66af67123215290b0e2c54",
        "git_object_id": "a65b392906e0c9aff2341820335b844c16e7ea6f",
        "schema_version": "review-record-v4",
        "round_id": "round-46-execute-operational-twin-candidate-lineage",
        "direction_id": "execute-operational-twin-candidate-lineage",
        "decision": "approved",
    },
    "implementation_notes": {
        "path": "orchestrator/rounds/round-46-execute-operational-twin-candidate-lineage/implementation-notes.md",
        "sha256": "c0f9b25f3680d3c475cdaaee8d3fa29d044b02b52f1e3b35548fe973d0247005",
        "git_object_id": "5b62a3957d1e27b877bee15bf6bc384617b555ac",
    },
    "blocker_summary": {
        "path": "orchestrator/rounds/round-46-execute-operational-twin-candidate-lineage/validation_logs/execution_blocker_summary.log",
        "sha256": "5d30b035639e4adecf0a04510c25faa6266e581f24a266571428a543167e0665",
        "git_object_id": "ad72e24ee59f37c2c38ffed88989de5d7dd2c498",
    },
    "operational_boundary": {
        "terminal_decision": "OPERATIONAL_TWIN_CANDIDATE_LINEAGE_EXECUTION_INCOMPLETE",
        "initial_attempt_count": 15,
        "failed_initial_attempt_count": 15,
        "materialized_attempt_count": 10,
        "prepublication_failure_count": 5,
        "evaluation_summary_count": 0,
        "recovery_attempt_count": 0,
        "scheduler_receipt_count": 0,
        "candidate_execution_catalog_created": False,
        "sufficiency_adjudicated": False,
        "scientific_twin_result": "NOT_RUN",
        "access_counters": {key: 0 for key in REQUIRED_ZERO_ACCESS_COUNTERS},
    },
}
CANDIDATE_LINEAGE_ROOT = Path(
    "Output/results/operational_twin/candidate_lineages"
)
CANDIDATE_FAILED_TERMINAL_KEYS = {
    "schema_version",
    "candidate_lineage_uid",
    "protocol_uid",
    "assignment_uid",
    "attempt_uid",
    "method_id",
    "seed",
    "output_root",
    "status",
    "failure_class",
    "terminal_reason",
    "error",
    "started_at",
    "finished_at",
    "scheduler_identity",
    "access_flags",
    "claim_flags",
    "fresh_initialization_receipt",
    "state_resume_used",
    "candidate_execution_authority",
    "candidate_resource_environment_receipt",
}


class OperationalTwinProtocolError(ValueError):
    """Raised when the frozen training protocol is not admissible."""



@dataclass(frozen=True)
class ResolvedCandidateExecutionProtocol:
    """Explicit authenticated inputs shared by the maintained execution path."""
    protocol: Mapping[str, object]
    candidate_manifest: Mapping[str, object]
    assignment_manifest: Mapping[str, object]
    command_catalog: Mapping[str, object]
    preflight: Mapping[str, object]
    execution: Mapping[str, object]
    handoff: Mapping[str, object]
    binding: Mapping[str, object]


def candidate_execution_binding_keys(keys: set[str], resolved_context: ResolvedCandidateExecutionProtocol | None) -> set[str]:
    if resolved_context is None:
        return set(keys)
    result = keys | {"successor_execution_binding", "runtime_dispatch_binding"}
    if "git_object_id" in result:
        result = (result - {"git_object_id"}) | {"size_bytes"}
    return result


def require_successor_execution_binding(value: Mapping[str, object], resolved_context: ResolvedCandidateExecutionProtocol | None) -> None:
    if resolved_context is not None and value.get("successor_execution_binding") != dict(resolved_context.binding):
        raise OperationalTwinProtocolError("candidate_successor_execution_binding_mismatch")


def _require_executing_commit(root: Path, commit: str, resolved_context: ResolvedCandidateExecutionProtocol | None) -> None:
    if resolved_context is None:
        return
    source = resolved_context.handoff["manifest"]["executing_source"]
    if subprocess.run(["git", "merge-base", "--is-ancestor", str(source["source_commit"]), commit],
                      cwd=root, capture_output=True, check=False).returncode:
        raise OperationalTwinProtocolError("candidate_unreviewed_execution_commit")
    # The handoff module imports this protocol, so resolve its reader locally.
    from Libs.utils.evaluation.operational_twin_successor_execution_handoff import _identities

    rows = source["source_files"]
    try:
        identities = _identities(root, commit, [str(row["path"]) for row in rows], current=False)
    except OperationalTwinProtocolError:
        raise
    except ValueError as error:
        raise OperationalTwinProtocolError("candidate_execution_commit_batch_malformed") from error
    if len(identities) != len(rows):
        raise OperationalTwinProtocolError("candidate_execution_commit_batch_malformed: row count")
    for row, identity in zip(rows, identities, strict=True):
        if identity["path"] != str(row["path"]):
            raise OperationalTwinProtocolError("candidate_execution_commit_batch_malformed: row order")
        if identity["sha256"] != row["sha256"]:
            raise OperationalTwinProtocolError("candidate_execution_commit_source_drift")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: object) -> str:
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str))


def _git_blob_bytes(root: Path, relative: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "show", f"HEAD:{relative}"],
            cwd=root,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise OperationalTwinProtocolError(
            f"tracked_authority_unavailable: {relative} is absent from the worktree and current Git tree."
        ) from error


def _git_object_bytes(root: Path, object_id: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "cat-file", "blob", object_id],
            cwd=root,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise OperationalTwinProtocolError(
            f"bound_git_object_unavailable: {object_id} cannot be read."
        ) from error


def _git_commit_path_bytes(root: Path, commit: str, relative: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "show", f"{commit}:{relative}"],
            cwd=root,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise OperationalTwinProtocolError(
            f"bound_commit_artifact_unavailable: {commit}:{relative} cannot be read."
        ) from error


def read_repo_bytes(root: Path, value: str | Path) -> bytes:
    """Read one declared repository file, falling back to its exact HEAD blob in sparse worktrees."""
    path = Path(value)
    if path.is_absolute():
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(root.resolve()).as_posix()
        except ValueError as error:
            raise OperationalTwinProtocolError(
                f"path_outside_repository: {resolved} is outside {root.resolve()}."
            ) from error
    else:
        relative = path.as_posix()
        resolved = (root / path).resolve()
    if resolved.is_file():
        return resolved.read_bytes()
    return _git_blob_bytes(root, relative)


def read_repo_json(root: Path, value: str | Path) -> dict[str, object]:
    payload = json.loads(read_repo_bytes(root, value).decode("utf-8"))
    if not isinstance(payload, dict):
        raise OperationalTwinProtocolError(f"json_root_not_object: {value}")
    return payload


def repo_file_identity(root: Path, value: str | Path) -> dict[str, object]:
    path = Path(value)
    relative = (
        path.resolve().relative_to(root.resolve()).as_posix()
        if path.is_absolute()
        else path.as_posix()
    )
    content = read_repo_bytes(root, relative)
    try:
        git_object_id = subprocess.check_output(
            ["git", "rev-parse", "--verify", f"HEAD:{relative}"],
            cwd=root,
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_object_id = None
    return {
        "path": relative,
        "sha256": hashlib.sha256(content).hexdigest(),
        "git_object_id": git_object_id,
    }


def candidate_runtime_artifact_identity(
    root: Path, value: str | Path,
    resolved_context: ResolvedCandidateExecutionProtocol | None,
) -> dict[str, object]:
    """Keep successor runtime byte identity stable across evidence publication."""
    if resolved_context is None:
        return repo_file_identity(root, value)
    # The retained-evidence module imports this protocol; defer the shared
    # content-identity helper until both modules are initialized.
    from Libs.utils.evaluation.operational_twin_candidate_evidence import artifact_identity

    path = Path(value)
    return artifact_identity(root, path if path.is_absolute() else root / path)


def _candidate_protocol(config: Mapping[str, object],
    resolved_context: ResolvedCandidateExecutionProtocol | None = None,
) -> dict[str, object]:
    if resolved_context is not None:
        if not isinstance(resolved_context, ResolvedCandidateExecutionProtocol):
            raise OperationalTwinProtocolError("candidate_resolved_context_type_invalid")
        return deepcopy(dict(resolved_context.protocol))
    protocol = config.get(CANDIDATE_PROTOCOL_KEY)
    if not isinstance(protocol, Mapping):
        raise OperationalTwinProtocolError(
            f"missing_candidate_protocol_config: {CANDIDATE_PROTOCOL_KEY} is absent."
        )
    return dict(protocol)


def candidate_lineage_identity_basis(protocol: Mapping[str, object]) -> dict[str, object]:
    return {
        key: deepcopy(protocol.get(key))
        for key in (
            "authoritative_date",
            "evidence_class",
            "parent_candidate_lineage_uid",
            "initialization",
            "scientific_contract",
            "historical_boundary",
            "sealed_manifest_metadata",
        )
    }


def candidate_protocol_identity_basis(protocol: Mapping[str, object]) -> dict[str, object]:
    basis = {
        key: deepcopy(value)
        for key, value in protocol.items()
        if key not in {"candidate_lineage_uid", "protocol_uid"}
    }
    basis["candidate_lineage_uid"] = compute_candidate_lineage_uid(protocol)
    return basis


def compute_candidate_lineage_uid(protocol: Mapping[str, object]) -> str:
    identity = dict(protocol.get("identity_policy", {}))
    prefix = str(identity.get("candidate_lineage_uid_prefix", ""))
    return f"{prefix}_{canonical_hash(candidate_lineage_identity_basis(protocol))[:24]}"


def compute_candidate_protocol_uid(protocol: Mapping[str, object]) -> str:
    identity = dict(protocol.get("identity_policy", {}))
    prefix = str(identity.get("protocol_uid_prefix", ""))
    return f"{prefix}_{canonical_hash(candidate_protocol_identity_basis(protocol))[:24]}"


def candidate_supersession_contract_hash(protocol: Mapping[str, object]) -> str:
    return canonical_hash(protocol.get("supersession_contract"))


def compute_candidate_assignment_uid(
    protocol: Mapping[str, object], method_id: str, seed: int
) -> str:
    identity = dict(protocol.get("identity_policy", {}))
    prefix = str(identity.get("assignment_uid_prefix", ""))
    digest = canonical_hash(
        {
            "candidate_lineage_uid": protocol.get("candidate_lineage_uid"),
            "method_id": method_id,
            "seed": int(seed),
        }
    )[:16]
    return f"{prefix}__{method_id}__seed_{int(seed)}__{digest}"


def compute_candidate_attempt_uid(
    protocol: Mapping[str, object], assignment_uid: str, attempt_kind: str
) -> str:
    template = str(dict(protocol.get("identity_policy", {})).get("attempt_uid_template", ""))
    return template.format(assignment_uid=assignment_uid, attempt_kind=attempt_kind)


def candidate_fresh_initialization_receipt(protocol: Mapping[str, object],
    resolved_context: ResolvedCandidateExecutionProtocol | None = None,
) -> dict[str, object]:
    if resolved_context is not None:
        return deepcopy(dict(resolved_context.candidate_manifest["initialization"]))
    initialization = dict(protocol.get("initialization", {}))
    return {
        "initialization_mode": initialization.get("initialization_mode"),
        "parent_candidate_lineage_uid": protocol.get("parent_candidate_lineage_uid"),
        "resume_allowed": initialization.get("resume_allowed"),
        **{field: initialization.get(field) for field in CANDIDATE_FORBIDDEN_STATE_INPUT_FIELDS},
        "historical_state_use_allowed": initialization.get("historical_state_use_allowed"),
    }


def candidate_identity_payload(
    assignment: Mapping[str, object], attempt: Mapping[str, object]
) -> dict[str, object]:
    return {
        "candidate_lineage_uid": assignment.get("candidate_lineage_uid"),
        "protocol_uid": assignment.get("protocol_uid"),
        "assignment_uid": assignment.get("assignment_uid"),
        "attempt_uid": attempt.get("attempt_uid"),
    }


def repo_path(root: Path, value: str | Path, *, must_exist: bool = False) -> Path:
    path = Path(value)
    resolved = path if path.is_absolute() else root / path
    return resolved.resolve(strict=must_exist)


def candidate_attempt_path(
    root: Path,
    value: str | Path,
    *,
    require_absent: bool,
) -> Path:
    """Validate one canonical candidate attempt path without following symlinks."""
    repository_root = root.resolve(strict=True)
    declared_value = os.fspath(value)
    if not declared_value:
        raise OperationalTwinProtocolError(
            "candidate_attempt_path_invalid: the declared attempt root is empty."
        )
    declared_path = Path(declared_value)
    if declared_path.is_absolute():
        raise OperationalTwinProtocolError(
            "candidate_attempt_path_absolute: candidate attempt roots must be repository-relative."
        )
    if ".." in declared_path.parts:
        raise OperationalTwinProtocolError(
            "candidate_attempt_path_parent_reference: candidate attempt roots must not contain '..'."
        )
    if declared_value != declared_path.as_posix():
        raise OperationalTwinProtocolError(
            "candidate_attempt_path_alias: candidate attempt roots must use one canonical POSIX spelling."
        )
    try:
        relative_attempt = declared_path.relative_to(CANDIDATE_LINEAGE_ROOT)
    except ValueError as error:
        raise OperationalTwinProtocolError(
            "candidate_attempt_path_outside_root: candidate attempt roots must be lexically contained under "
            f"{CANDIDATE_LINEAGE_ROOT.as_posix()}."
        ) from error
    if not relative_attempt.parts:
        raise OperationalTwinProtocolError(
            "candidate_attempt_path_invalid: the candidate-lineage root is not an attempt leaf."
        )

    current = repository_root
    for component in declared_path.parts[:-1]:
        current = current / component
        if not os.path.lexists(current):
            continue
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise OperationalTwinProtocolError(
                "candidate_attempt_path_symlink: candidate attempt ancestors must not be symlinks."
            )
        if not stat.S_ISDIR(metadata.st_mode):
            raise OperationalTwinProtocolError(
                "candidate_attempt_path_ancestor_not_directory: candidate attempt ancestors must be directories."
            )

    attempt_path = repository_root.joinpath(*declared_path.parts)
    if os.path.lexists(attempt_path):
        metadata = attempt_path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise OperationalTwinProtocolError(
                "candidate_attempt_path_symlink: the candidate attempt leaf must not be a symlink."
            )
        if require_absent:
            raise OperationalTwinProtocolError(
                "candidate_attempt_output_collision: the manifest-bound attempt root already exists."
            )
    return attempt_path


def relative_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _block(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def build_assignments(protocol: Mapping[str, object], root: Path) -> list[dict[str, object]]:
    methods = list(protocol.get("methods", []))
    seeds = [int(seed) for seed in protocol.get("seeds", [])]
    assignment_root = Path(str(dict(protocol.get("output_roots", {})).get("assignment_root", "")))
    assignments: list[dict[str, object]] = []
    for method in methods:
        method_id = str(dict(method).get("method_id", ""))
        for seed in seeds:
            assignment_uid = f"operational_twin__{method_id}__seed_{seed}"
            output_root = assignment_root / method_id / f"seed_{seed}"
            assignments.append(
                {
                    "assignment_uid": assignment_uid,
                    "method_id": method_id,
                    "seed": seed,
                    "output_root": output_root.as_posix(),
                    "immutable_output_root": True,
                    "attempt_retention": "retain_all_attempts_under_assignment_uid",
                    "execution_contract": dict(method).get("execution_contract"),
                    "metrics_contract": dict(method).get("metrics_contract"),
                    "runtime_contract": METHOD_EXECUTION_CONTRACTS.get(method_id, {}),
                    "resource_profile": "gpu_single" if method_id != "persistence_no_action" else "cpu_light",
                }
            )
    return assignments


def resolve_method_execution_contract(method_id: str, execution_contract: str | None = None) -> dict[str, object]:
    contract = METHOD_EXECUTION_CONTRACTS.get(method_id)
    if contract is None:
        raise OperationalTwinProtocolError(f"unsupported_method_id: {method_id} is not one of the five frozen methods.")
    if execution_contract is not None and str(execution_contract) != str(contract["execution_contract"]):
        raise OperationalTwinProtocolError(
            f"method_execution_contract_mismatch: {method_id} expected {contract['execution_contract']}, received {execution_contract}."
        )
    return dict(contract)


def validate_protocol_config(config: Mapping[str, object], root: Path) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    protocol = config.get("operational_twin_training_protocol")
    if not isinstance(protocol, Mapping):
        return [_block("missing_protocol_config", "operational_twin_training_protocol is absent from the config stack.")]
    methods = [str(dict(row).get("method_id", "")) for row in protocol.get("methods", [])]
    seeds = [int(seed) for seed in protocol.get("seeds", [])] if isinstance(protocol.get("seeds"), list) else []
    if methods != REQUIRED_METHOD_IDS:
        blockers.append(_block("method_identity_mismatch", "The protocol must declare exactly the five fixed method identities in order."))
    for method in protocol.get("methods", []):
        row = dict(method)
        try:
            expected_contract = resolve_method_execution_contract(str(row.get("method_id", "")), str(row.get("execution_contract", "")))
            field_pairs = {
                "runtime_family": "runtime_family",
                "action_conditioning": "action_conditioning",
                "loss_contract": "loss_name",
                "calibration_eligible": "calibration_eligible",
                "checkpoint_creation_allowed": "checkpoint_creation_allowed",
            }
            for yaml_field, contract_field in field_pairs.items():
                if row.get(yaml_field) != expected_contract[contract_field]:
                    blockers.append(_block("method_runtime_field_mismatch", f"{row.get('method_id')} {yaml_field} does not match the frozen runtime contract."))
        except OperationalTwinProtocolError as error:
            blockers.append(_block("method_execution_contract_invalid", str(error)))
    if len(seeds) != 3 or len(set(seeds)) != 3:
        blockers.append(_block("seed_policy_mismatch", "The protocol must declare exactly three unique integer seeds."))
    seed_policy = dict(protocol.get("seed_policy", {}))
    if seed_policy.get("hidden_extra_seed_lists_allowed") is not False:
        blockers.append(_block("hidden_seed_policy_open", "Hidden extra seed lists must be explicitly forbidden."))
    for hidden_key in ("extra_seeds", "candidate_seeds", "replacement_seeds", "selection_seeds", "rerun_seeds"):
        if hidden_key in protocol:
            blockers.append(_block("hidden_seed_list_present", f"Forbidden seed list is present: {hidden_key}."))
    assignments = build_assignments(protocol, root)
    assignment_uids = [str(row["assignment_uid"]) for row in assignments]
    output_roots = [str(row["output_root"]) for row in assignments]
    if len(assignments) != 15 or len(set(assignment_uids)) != 15:
        blockers.append(_block("assignment_count_mismatch", "The protocol must yield exactly 15 unique method-seed assignments."))
    if len(set(output_roots)) != len(output_roots):
        blockers.append(_block("assignment_output_collision", "Assignment output roots must be disjoint."))
    catalog_root = str(dict(protocol.get("output_roots", {})).get("catalog_root", ""))
    if catalog_root in output_roots or any(path.startswith(f"{catalog_root}/") for path in output_roots):
        blockers.append(_block("catalog_writer_collision", "The single-writer catalog path cannot contain an assignment output root."))
    for label, value in dict(protocol.get("manifests", {})).items():
        try:
            lowered = Path(str(value)).as_posix().lower()
            if any(token in lowered for token in FORBIDDEN_PATH_TOKENS):
                raise ValueError("Forbidden sealed, protected, test, or six-context path token was rejected before content access.")
            reject_forbidden_path_tokens(value, label=f"{label} manifest")
        except ValueError as error:
            blockers.append(_block("forbidden_manifest_path", str(error)))
    for label, value in dict(protocol.get("output_roots", {})).items():
        if label in {"preflight_authority", "preflight_report"}:
            continue
        try:
            lowered = Path(str(value)).as_posix().lower()
            if any(token in lowered for token in FORBIDDEN_PATH_TOKENS):
                raise ValueError("Forbidden sealed, protected, test, or six-context path token was rejected before content access.")
            reject_forbidden_path_tokens(value, label=f"{label} output")
        except ValueError as error:
            blockers.append(_block("forbidden_output_path", str(error)))
    for key in REQUIRED_ZERO_ACCESS_COUNTERS:
        if dict(protocol.get("access_counters", {})).get(key) != 0:
            blockers.append(_block("nonzero_access_counter", f"{key} must remain zero."))
    closed_flags = dict(protocol.get("closed_flags", {}))
    for key in ("sufficient_training", "twin_qualification_pass", "formal_qualification"):
        if closed_flags.get(key) is not False:
            blockers.append(_block("closed_flag_open", f"{key} must remain false."))
    if dict(protocol.get("selection_and_calibration", {})).get("checkpoint_selection_role") != "development_only":
        blockers.append(_block("selection_role_invalid", "Checkpoint/model selection must be development-only."))
    if dict(protocol.get("selection_and_calibration", {})).get("uncertainty_calibration_role") != "calibration_worlds_only":
        blockers.append(_block("calibration_role_invalid", "UQ calibration must be calibration-world-only."))
    return blockers


def validate_open_data_binding(config: Mapping[str, object], root: Path) -> tuple[list[dict[str, str]], dict[str, object]]:
    blockers: list[dict[str, str]] = []
    protocol = dict(config.get("operational_twin_training_protocol", {}))
    authority_path = repo_path(root, str(protocol.get("open_data_authority", "")), must_exist=False)
    if not authority_path.is_file():
        return [_block("missing_open_data_qualification", f"Open-data qualification is missing: {relative_path(root, authority_path)}.")], {}
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    if authority.get("terminal_decision") != QUALIFIED_TERMINAL or authority.get("training_authorized") is not True:
        blockers.append(_block("open_data_not_qualified", "Open-data authority is not positive and training-authorized."))
    if authority.get("sufficient_training") is not False or authority.get("twin_qualification_pass") is not False or authority.get("formal_qualification") is not False:
        blockers.append(_block("downstream_gate_open", "Open-data authority must not assert sufficient training, Twin pass, or formal qualification."))
    counters = dict(authority.get("forbidden_access_counters", {}))
    sealed = dict(authority.get("sealed_material_access", {}))
    if any(value != 0 for value in counters.values()) or any(value != 0 for value in sealed.values()):
        blockers.append(_block("open_data_nonzero_access", "Open-data authority reports nonzero forbidden or sealed access."))
    protocol_manifests = dict(protocol.get("manifests", {}))
    authority_paths = dict(authority.get("input_identities", {}))
    for role in ("training", "development", "calibration"):
        configured = str(protocol_manifests.get(role, ""))
        if dict(authority_paths.get(role, {})).get("path") != configured:
            blockers.append(_block("manifest_identity_mismatch", f"{role} manifest does not match the open-data authority."))
    return blockers, authority


def validate_resource_receipt(receipt: Mapping[str, object], *, config_hash: str) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    if receipt.get("schema_version") != "neuronsifter-operational-twin-resource-receipt-v1":
        blockers.append(_block("resource_receipt_schema_mismatch", "Resource receipt schema_version is invalid."))
    if receipt.get("config_hash") != config_hash:
        blockers.append(_block("resource_receipt_config_hash_mismatch", "Resource receipt does not bind the frozen protocol config hash."))
    for probe_id in ("cpu_probe", "gpu_probe"):
        probe = dict(dict(receipt.get("probes", {})).get(probe_id, {}))
        if probe.get("exit_status") != 0:
            blockers.append(_block("resource_probe_failed", f"{probe_id} did not exit successfully."))
        if not str(probe.get("job_name", "")).startswith("NrnSift_"):
            blockers.append(_block("resource_probe_job_name_invalid", f"{probe_id} does not use the NrnSift_ prefix."))
        sacct = str(probe.get("sacct_evidence", ""))
        if "COMPLETED" not in sacct or "|0:0|" not in sacct:
            blockers.append(_block("resource_probe_sacct_not_terminal", f"{probe_id} lacks terminal successful sacct evidence."))
        job_id = str(probe.get("job_id", ""))
        stdout_path = str(probe.get("stdout_path", ""))
        stdout_sha256 = str(probe.get("stdout_sha256", ""))
        if probe_id == "gpu_probe":
            stdout_text = str(dict(probe.get("environment", {})).get("probe_stdout", ""))
            if f"SLURM_JOB_ID={job_id}" not in stdout_text:
                blockers.append(_block("resource_probe_stdout_job_id_mismatch", "GPU probe stdout does not contain the bound Slurm job id."))
            if f"|{job_id}|" not in f"|{sacct}" and not sacct.startswith(f"{job_id}|"):
                blockers.append(_block("resource_probe_sacct_job_id_mismatch", "GPU probe sacct evidence does not contain the bound Slurm job id."))
            if "CUDA_VISIBLE_DEVICES=" not in stdout_text or "cuda_available" not in stdout_text:
                blockers.append(_block("resource_probe_stdout_environment_incomplete", "GPU probe stdout must include allocated CUDA devices and Torch/CUDA information."))
        if stdout_path and stdout_sha256:
            resolved_stdout = Path(stdout_path)
            if not resolved_stdout.is_absolute():
                resolved_stdout = Path.cwd() / resolved_stdout
            if resolved_stdout.is_file() and sha256_file(resolved_stdout) != stdout_sha256:
                blockers.append(_block("resource_probe_stdout_hash_mismatch", f"{probe_id} stdout hash does not match its committed evidence file."))
    return blockers


def build_assignment_manifest(config: Mapping[str, object], root: Path) -> dict[str, object]:
    protocol = dict(config["operational_twin_training_protocol"])
    assignments = build_assignments(protocol, root)
    payload = {
        "schema_version": "neuronsifter-operational-twin-assignment-manifest-v1",
        "protocol_uid": protocol["protocol_uid"],
        "method_ids": REQUIRED_METHOD_IDS,
        "seeds": [int(seed) for seed in protocol["seeds"]],
        "assignment_count": len(assignments),
        "assignments": assignments,
        "single_writer_catalog_root": dict(protocol["output_roots"])["catalog_root"],
        "hidden_seed_policy": dict(protocol["seed_policy"]),
        "access_counters": dict(protocol["access_counters"]),
    }
    payload["assignment_manifest_hash"] = canonical_hash({k: v for k, v in payload.items() if k != "assignment_manifest_hash"})
    return payload


def build_resource_receipt(config: Mapping[str, object], root: Path, *, sinfo_text: str, sacct_text: str, probe_outputs: Mapping[str, object]) -> dict[str, object]:
    protocol_text = yaml.safe_dump(config["operational_twin_training_protocol"], sort_keys=True)
    return {
        "schema_version": "neuronsifter-operational-twin-resource-receipt-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_prefix": (root / "PROJECT_PREFIX").read_text(encoding="utf-8").strip(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "config_hash": sha256_text(protocol_text),
        "control_plane": {"hostname": socket.gethostname(), "platform": platform.platform()},
        "sinfo_evidence": sinfo_text,
        "probes": dict(probe_outputs),
        "sacct_evidence": sacct_text,
        "training_executed": False,
        "calibration_fitting_executed": False,
        "checkpoint_created": False,
    }


def build_preflight_authority(config: Mapping[str, object], root: Path, assignment_manifest: Mapping[str, object], resource_receipt: Mapping[str, object], open_authority: Mapping[str, object]) -> dict[str, object]:
    protocol = dict(config["operational_twin_training_protocol"])
    protocol_text = yaml.safe_dump(protocol, sort_keys=True)
    return {
        "schema_version": "neuronsifter-operational-twin-training-protocol-preflight-v1",
        "status": "PASS",
        "terminal_decision": "OPERATIONAL_TWIN_TRAINING_PROTOCOL_PREFLIGHT_PASSED",
        "authoritative_date": protocol["authoritative_date"],
        "protocol_uid": protocol["protocol_uid"],
        "config_hash": sha256_text(protocol_text),
        "open_data_terminal": open_authority.get("terminal_decision"),
        "open_data_training_authorized": open_authority.get("training_authorized"),
        "input_content_identities": open_authority.get("input_content_identities"),
        "method_ids": REQUIRED_METHOD_IDS,
        "seeds": [int(seed) for seed in protocol["seeds"]],
        "assignment_count": assignment_manifest["assignment_count"],
        "assignment_manifest_hash": assignment_manifest["assignment_manifest_hash"],
        "resource_receipt_hash": canonical_hash(resource_receipt),
        "output_roots": dict(protocol["output_roots"]),
        "statistics": dict(protocol["statistics"]),
        "failure_policy": dict(protocol["failure_policy"]),
        "selection_and_calibration": dict(protocol["selection_and_calibration"]),
        "access_counters": dict(protocol["access_counters"]),
        "closed_flags": dict(protocol["closed_flags"]),
        "sufficient_training": False,
        "twin_qualification_pass": False,
        "formal_qualification": False,
        "training_executed": False,
        "calibration_fitting_executed": False,
        "checkpoint_created": False,
        "sealed_material_access": 0,
        "tool_mcp_agent_work": "closed",
    }


def validate_preflight_authority(config: Mapping[str, object], root: Path, preflight: Mapping[str, object], assignment_manifest: Mapping[str, object], resource_receipt: Mapping[str, object]) -> list[dict[str, str]]:
    blockers = validate_protocol_config(config, root)
    open_blockers, open_authority = validate_open_data_binding(config, root)
    blockers.extend(open_blockers)
    protocol_text = yaml.safe_dump(config["operational_twin_training_protocol"], sort_keys=True)
    config_hash = sha256_text(protocol_text)
    blockers.extend(validate_resource_receipt(resource_receipt, config_hash=config_hash))
    expected_manifest = build_assignment_manifest(config, root)
    if assignment_manifest != expected_manifest:
        blockers.append(_block("assignment_manifest_drift", "Assignment manifest does not match the frozen protocol config."))
    if preflight.get("status") != "PASS" or preflight.get("terminal_decision") != "OPERATIONAL_TWIN_TRAINING_PROTOCOL_PREFLIGHT_PASSED":
        blockers.append(_block("preflight_not_passed", "Preflight authority is not terminal PASS."))
    if preflight.get("config_hash") != config_hash:
        blockers.append(_block("preflight_config_hash_mismatch", "Preflight authority config hash drifted."))
    if preflight.get("assignment_manifest_hash") != expected_manifest["assignment_manifest_hash"]:
        blockers.append(_block("preflight_assignment_hash_mismatch", "Preflight assignment hash drifted."))
    if preflight.get("resource_receipt_hash") != canonical_hash(resource_receipt):
        blockers.append(_block("preflight_resource_hash_mismatch", "Preflight resource receipt hash drifted."))
    if open_authority and preflight.get("input_content_identities") != open_authority.get("input_content_identities"):
        blockers.append(_block("preflight_open_identity_mismatch", "Preflight open-data identities drifted."))
    return blockers


def require_assignment_preflight(config: Mapping[str, object], root: Path, *, assignment_uid: str, method_id: str, seed: int, output_root: str | Path) -> dict[str, object]:
    protocol = dict(config.get("operational_twin_training_protocol", {}))
    paths = dict(protocol.get("output_roots", {}))
    manifest_path = repo_path(root, paths.get("assignment_manifest", ""), must_exist=True)
    receipt_path = repo_path(root, paths.get("resource_receipt", ""), must_exist=True)
    preflight_path = repo_path(root, paths.get("preflight_authority", ""), must_exist=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    blockers = validate_preflight_authority(config, root, preflight, manifest, receipt)
    matching = [row for row in manifest.get("assignments", []) if row.get("assignment_uid") == assignment_uid]
    if len(matching) != 1:
        blockers.append(_block("assignment_uid_invalid", "Requested assignment UID is not present exactly once in the frozen manifest."))
    else:
        assignment = dict(matching[0])
        if assignment.get("method_id") != method_id or int(assignment.get("seed")) != int(seed):
            blockers.append(_block("assignment_identity_mismatch", "Requested method or seed does not match assignment UID."))
        try:
            resolve_method_execution_contract(str(assignment.get("method_id")), str(assignment.get("execution_contract")))
        except OperationalTwinProtocolError as error:
            blockers.append(_block("assignment_method_contract_invalid", str(error)))
        expected_output = repo_path(root, str(assignment["output_root"]), must_exist=False)
        supplied_output = repo_path(root, output_root, must_exist=False)
        if supplied_output != expected_output:
            blockers.append(_block("assignment_output_root_mismatch", "Output root does not match the frozen immutable assignment root."))
    if blockers:
        raise OperationalTwinProtocolError("; ".join(f"{row['code']}: {row['message']}" for row in blockers))
    return matching[0]


def compute_candidate_preflight_uid(protocol: Mapping[str, object]) -> str:
    digest = canonical_hash(
        {
            "candidate_lineage_uid": protocol.get("candidate_lineage_uid"),
            "protocol_uid": protocol.get("protocol_uid"),
            "authoritative_date": protocol.get("authoritative_date"),
        }
    )[:24]
    return f"operational_twin_candidate_lineage_preflight_{digest}"


def _candidate_scientific_contract_from_v1(config: Mapping[str, object]) -> dict[str, object]:
    legacy = config.get("operational_twin_training_protocol")
    if not isinstance(legacy, Mapping):
        raise OperationalTwinProtocolError(
            "missing_historical_protocol_config: operational_twin_training_protocol is absent."
        )
    return {key: deepcopy(legacy.get(key)) for key in CANDIDATE_SCIENTIFIC_EQUALITY_FIELDS}


def _require_exact_keys(
    value: object,
    expected: set[str],
    *,
    label: str,
    blockers: list[dict[str, str]],
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        blockers.append(_block("candidate_schema_mapping_required", f"{label} must be a mapping."))
        return {}
    row = dict(value)
    actual = set(row)
    if actual != expected:
        blockers.append(
            _block(
                "candidate_schema_fields_mismatch",
                f"{label} fields differ; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}.",
            )
        )
    return row


def _validate_candidate_exact_section(
    schema: Mapping[str, object],
    section_name: str,
    value: object,
    *,
    blockers: list[dict[str, str]],
    error_code: str,
    field_error_codes: Mapping[str, str] | None = None,
) -> bool:
    """Validate one exact candidate section from its embedded Draft 2020-12 schema."""
    section_schema_value = schema.get(section_name)
    if not isinstance(section_schema_value, Mapping):
        blockers.append(
            _block(
                "candidate_schema_invalid",
                f"Schema section {section_name} must be a Draft 2020-12 mapping.",
            )
        )
        return False
    section_schema = dict(section_schema_value)
    try:
        Draft202012Validator.check_schema(section_schema)
    except SchemaError as error:
        blockers.append(
            _block(
                "candidate_schema_invalid",
                f"Schema section {section_name} is invalid: {error.message}",
            )
        )
        return False
    errors = sorted(
        Draft202012Validator(section_schema).iter_errors(value),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if not errors:
        return True
    details = []
    matched_codes = [error_code]
    field_codes = dict(field_error_codes or {})
    for error in errors:
        path = [str(part) for part in error.absolute_path]
        location = ".".join([section_name, *path])
        details.append(f"{location}: {error.message}")
        if path and path[0] in field_codes:
            matched_codes.append(field_codes[path[0]])
    message = "; ".join(details)
    for code in dict.fromkeys(matched_codes):
        blockers.append(_block(code, message))
    return False


def _candidate_execution_contract(protocol: Mapping[str, object],
    resolved_context: ResolvedCandidateExecutionProtocol | None = None,
) -> dict[str, object]:
    value = protocol.get("execution_authority_contract")
    if not isinstance(value, Mapping):
        raise OperationalTwinProtocolError(
            "candidate_execution_authority_contract_missing: "
            "execution_authority_contract is absent."
        )
    result = dict(value)
    if resolved_context is not None:
        result.update(resolved_context.execution["binding_contract"]["authority_overrides"])
    return result


def candidate_execution_artifact_paths(
    protocol: Mapping[str, object],
    assignment: Mapping[str, object],
    attempt: Mapping[str, object],
) -> dict[str, str]:
    """Return the manifest-derived, per-attempt authority paths."""
    contract = _candidate_execution_contract(protocol)
    values = {
        "candidate_lineage_uid": assignment.get("candidate_lineage_uid"),
        "assignment_uid": assignment.get("assignment_uid"),
        "attempt_uid": attempt.get("attempt_uid"),
    }
    return {
        "execution_authority": str(
            contract["candidate_execution_authority_path_template"]
        ).format(**values),
        "resource_environment_receipt": str(
            contract["resource_environment_receipt_path_template"]
        ).format(**values),
    }


def candidate_source_identity_hash(preflight: Mapping[str, object]) -> str:
    return canonical_hash(
        {
            "config_identity": preflight.get("config_identity"),
            "schema_identity": preflight.get("schema_identity"),
            "source_identities": preflight.get("source_identities"),
        }
    )


def compute_candidate_resource_receipt_uid(receipt: Mapping[str, object]) -> str:
    payload = deepcopy(dict(receipt))
    payload.pop("receipt_uid", None)
    return f"operational_twin_candidate_resource_receipt_{canonical_hash(payload)[:24]}"


def compute_candidate_execution_authority_uid(authority: Mapping[str, object]) -> str:
    payload = deepcopy(dict(authority))
    payload.pop("authority_uid", None)
    return f"operational_twin_candidate_execution_authority_{canonical_hash(payload)[:24]}"


def compute_candidate_calibration_authority_uid(
    authority: Mapping[str, object],
) -> str:
    payload = deepcopy(dict(authority))
    payload.pop("authority_uid", None)
    payload.pop("created_at", None)
    return (
        "operational_twin_candidate_calibration_authority_"
        f"{canonical_hash(payload)[:24]}"
    )


def _parse_utc_timestamp(value: object, *, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise OperationalTwinProtocolError(
            f"candidate_execution_timestamp_invalid: {label} is not ISO-8601."
        ) from error
    if parsed.tzinfo is None:
        raise OperationalTwinProtocolError(
            f"candidate_execution_timestamp_invalid: {label} must be timezone-aware."
        )
    return parsed.astimezone(timezone.utc)


def _current_git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise OperationalTwinProtocolError(
            "candidate_execution_source_commit_unavailable: current Git commit cannot be resolved."
        ) from error


def candidate_runtime_slurm_identity(
    *,
    hostname: str | None = None,
    values: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Capture raw Slurm fields and derive one authoritative resource-query ID."""

    raw_keys = CANDIDATE_RESOURCE_SLURM_KEYS - {"resource_query_job_id"}
    if values is None:
        identity: dict[str, object] = {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
            "slurm_job_partition": os.environ.get("SLURM_JOB_PARTITION"),
            "slurm_job_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
            "slurm_cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "slurm_mem_per_node": os.environ.get("SLURM_MEM_PER_NODE"),
            "slurm_mem_per_cpu": os.environ.get("SLURM_MEM_PER_CPU"),
            "slurm_job_gres": os.environ.get("SLURM_JOB_GRES"),
            "hostname": hostname or socket.gethostname().split(".", 1)[0],
            "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
            "slurm_gpus_on_node": os.environ.get("SLURM_GPUS_ON_NODE"),
            "slurm_step_gpus": os.environ.get("SLURM_STEP_GPUS"),
        }
        declared_query_id = None
    else:
        supplied_keys = set(values)
        if (
            supplied_keys != raw_keys
            and supplied_keys != CANDIDATE_RESOURCE_SLURM_KEYS
        ):
            raise OperationalTwinProtocolError(
                "candidate_runtime_slurm_identity_invalid: Slurm identity keyset is not exact."
            )
        identity = {key: values.get(key) for key in raw_keys}
        declared_query_id = values.get("resource_query_job_id")

    job_id = str(identity.get("slurm_job_id") or "").strip()
    array_job_id = str(identity.get("slurm_array_job_id") or "").strip()
    array_task_id = str(identity.get("slurm_array_task_id") or "").strip()
    if not job_id:
        raise OperationalTwinProtocolError(
            "candidate_runtime_slurm_identity_invalid: SLURM_JOB_ID is required."
        )
    array_job_present = bool(array_job_id)
    array_task_present = bool(array_task_id)
    if array_job_present != array_task_present:
        raise OperationalTwinProtocolError(
            "candidate_runtime_slurm_identity_invalid: array job and task identities must be present together."
        )
    if array_job_present:
        if not re.fullmatch(r"[1-9][0-9]*", array_job_id) or not re.fullmatch(
            r"[0-9]+", array_task_id
        ):
            raise OperationalTwinProtocolError(
                "candidate_runtime_slurm_identity_invalid: array job or task identity is malformed."
            )
        resource_query_job_id = f"{array_job_id}_{array_task_id}"
        if not (
            re.fullmatch(r"[1-9][0-9]*", job_id)
            or job_id == resource_query_job_id
        ):
            raise OperationalTwinProtocolError(
                "candidate_runtime_slurm_identity_invalid: SLURM_JOB_ID contradicts the array-element identity."
            )
    else:
        if not re.fullmatch(r"[1-9][0-9]*", job_id):
            raise OperationalTwinProtocolError(
                "candidate_runtime_slurm_identity_invalid: non-array SLURM_JOB_ID is malformed."
            )
        resource_query_job_id = job_id
    if declared_query_id is not None and str(declared_query_id) != resource_query_job_id:
        raise OperationalTwinProtocolError(
            "candidate_runtime_slurm_identity_invalid: derived resource-query identity drifted."
        )
    identity["resource_query_job_id"] = resource_query_job_id
    return identity


def _runtime_environment_identity(
    *,
    hostname: str,
    cuda_available: bool,
    cuda_device_count: int,
    torch_version: str,
) -> dict[str, object]:
    return {
        "hostname": hostname,
        "python_version": platform.python_version(),
        "torch_version": torch_version,
        "cuda_available": bool(cuda_available),
        "cuda_device_count": int(cuda_device_count),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _memory_mebibytes(value: object) -> int | None:
    text = str(value or "").strip().upper()
    if not text:
        return None
    multipliers = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}
    suffix = text[-1]
    if suffix in multipliers:
        number = text[:-1]
        multiplier = multipliers[suffix]
    else:
        number = text
        multiplier = 1
    try:
        parsed = float(number) * multiplier
    except ValueError:
        return None
    if parsed <= 0 or not parsed.is_integer():
        return None
    return int(parsed)


def _scontrol_fields(stdout: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in stdout.replace("\n", " ").split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        fields.setdefault(key, value)
    return fields


def _tres_fields(value: object) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in str(value or "").split(","):
        if "=" not in token:
            continue
        key, item = token.split("=", 1)
        fields[key] = item
    return fields


def _strict_scontrol_fields(stdout: str) -> dict[str, str]:
    """Parse exactly one Slurm oneliner and reject ambiguous contract fields."""

    records = [line.strip() for line in stdout.splitlines() if line.strip()]
    if len(records) != 1:
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_resource_invalid: scontrol must contain exactly one job record."
        )
    contract_fields = {
        "JobId",
        "ArrayJobId",
        "ArrayTaskId",
        "Partition",
        "NodeList",
        "CPUs/Task",
        "TRES",
        "MinMemoryNode",
        "OverSubscribe",
        "TresPerNode",
        *CANDIDATE_FORBIDDEN_LEGACY_SCHEDULER_FIELDS,
    }
    fields: dict[str, str] = {}
    for token in records[0].split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key not in contract_fields:
            continue
        if key in fields:
            raise OperationalTwinProtocolError(
                f"candidate_resource_receipt_resource_invalid: duplicate scontrol field {key}."
            )
        if not value:
            raise OperationalTwinProtocolError(
                f"candidate_resource_receipt_resource_invalid: empty scontrol field {key}."
            )
        fields[key] = value
    return fields


def _strict_tres_fields(value: object, *, label: str) -> dict[str, str]:
    text = str(value or "")
    if not text:
        raise OperationalTwinProtocolError(
            f"candidate_resource_receipt_resource_invalid: {label} is empty."
        )
    fields: dict[str, str] = {}
    for token in text.split(","):
        if token.count("=") != 1:
            raise OperationalTwinProtocolError(
                f"candidate_resource_receipt_resource_invalid: malformed {label} entry."
            )
        key, item = token.split("=", 1)
        if not key or not item or key in fields:
            raise OperationalTwinProtocolError(
                f"candidate_resource_receipt_resource_invalid: ambiguous {label} entry {key!r}."
            )
        fields[key] = item
    return fields


def _require_candidate_resource_evidence_v1(
    receipt: Mapping[str, object],
    requirements: Mapping[str, object],
    resource_command_template: object,
) -> None:
    """Retain the historical Round 43 parser for explicit v1 replay only."""

    resource_stdout = receipt.get("resource_stdout")
    if not isinstance(resource_stdout, str) or not resource_stdout.strip():
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_resource_invalid: scontrol resource evidence is empty."
        )
    slurm = dict(receipt.get("slurm", {}))
    job_id = str(slurm.get("slurm_job_id") or "")
    expected_command = [
        str(value).format(slurm_job_id=job_id)
        for value in resource_command_template
    ]
    if receipt.get("resource_command") != expected_command:
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_resource_invalid: scontrol resource command drifted."
        )
    fields = _scontrol_fields(resource_stdout)
    if (
        not job_id
        or fields.get("JobId") != job_id
        or fields.get("Partition") != str(requirements.get("partition"))
        or fields.get("NodeList") != str(requirements.get("nodelist"))
        or fields.get("CPUs/Task") != str(requirements.get("cpus_per_task"))
    ):
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_resource_mismatch: scontrol job, node, partition, or CPU evidence drifted."
        )
    expected_memory = _memory_mebibytes(requirements.get("memory"))
    expected_gres = requirements.get("gres")
    for field_name in ("ReqTRES", "AllocTRES"):
        tres = _tres_fields(fields.get(field_name))
        if (
            tres.get("cpu") != str(requirements.get("cpus_per_task"))
            or _memory_mebibytes(tres.get("mem")) != expected_memory
        ):
            raise OperationalTwinProtocolError(
                "candidate_resource_receipt_resource_mismatch: scontrol CPU or memory TRES evidence drifted."
            )
        gpu_count = tres.get("gres/gpu")
        if expected_gres == "gpu:1" and gpu_count != "1":
            raise OperationalTwinProtocolError(
                "candidate_resource_receipt_gres_mismatch: scontrol GPU GRES evidence drifted."
            )
        if expected_gres is None and gpu_count is not None:
            raise OperationalTwinProtocolError(
                "candidate_resource_receipt_gres_mismatch: CPU-light execution unexpectedly requested or allocated GPU GRES."
            )
    shared = fields.get("Shared")
    if requirements.get("exclusive") is False and shared not in {
        "OK",
        "YES",
        "USER",
        "MCS",
    }:
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_exclusivity_mismatch: non-exclusive execution is not proven by scontrol."
        )
    if requirements.get("exclusive") is True and shared != "NO":
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_exclusivity_mismatch: exclusive execution is not proven by scontrol."
        )


def _require_candidate_resource_evidence_v2(
    receipt: Mapping[str, object],
    requirements: Mapping[str, object],
    resource_command_template: object,
) -> None:
    """Validate the strict Slurm 22.05.9 receipt mapping frozen in Round 45."""

    resource_stdout = receipt.get("resource_stdout")
    if not isinstance(resource_stdout, str) or not resource_stdout.strip():
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_resource_invalid: scontrol resource evidence is empty."
        )
    slurm = dict(receipt.get("slurm", {}))
    job_id = str(slurm.get("slurm_job_id") or "")
    expected_command = [
        str(value).format(slurm_job_id=job_id)
        for value in resource_command_template
    ]
    if receipt.get("resource_command") != expected_command:
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_resource_invalid: scontrol resource command drifted."
        )
    fields = _strict_scontrol_fields(resource_stdout)
    forbidden = [
        field
        for field in CANDIDATE_FORBIDDEN_LEGACY_SCHEDULER_FIELDS
        if field in fields
    ]
    if forbidden:
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_legacy_field_forbidden: v2 cannot consume "
            + ", ".join(forbidden)
            + "."
        )
    required_fields = {
        "JobId",
        "Partition",
        "NodeList",
        "CPUs/Task",
        "TRES",
        "MinMemoryNode",
        "OverSubscribe",
    }
    missing = sorted(required_fields - set(fields))
    if missing:
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_resource_invalid: missing scontrol fields "
            + ", ".join(missing)
            + "."
        )
    if (
        not job_id
        or fields["JobId"] != job_id
        or fields["Partition"] != str(requirements.get("partition"))
        or fields["NodeList"] != str(requirements.get("nodelist"))
        or fields["CPUs/Task"] != str(requirements.get("cpus_per_task"))
    ):
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_resource_mismatch: scontrol job, node, partition, or CPU evidence drifted."
        )
    tres = _strict_tres_fields(fields["TRES"], label="TRES")
    if tres.get("cpu") != str(requirements.get("cpus_per_task")):
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_resource_mismatch: TRES.cpu differs from the frozen CPU allocation."
        )
    memory = fields["MinMemoryNode"].upper()
    if not re.fullmatch(r"[1-9][0-9]*[KMGT]", memory):
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_resource_invalid: MinMemoryNode has a malformed unit."
        )
    if _memory_mebibytes(memory) != _memory_mebibytes(requirements.get("memory")):
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_resource_mismatch: MinMemoryNode differs from the frozen memory allocation."
        )
    if fields["OverSubscribe"] != "OK" or requirements.get("exclusive") is not False:
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_exclusivity_mismatch: OverSubscribe=OK is required for the frozen non-exclusive profile."
        )
    expected_gres = requirements.get("gres")
    if expected_gres == "gpu:1":
        if fields.get("TresPerNode") != "gres:gpu:1":
            raise OperationalTwinProtocolError(
                "candidate_resource_receipt_gres_mismatch: gpu_single requires exactly TresPerNode=gres:gpu:1."
            )
    elif "TresPerNode" in fields:
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_gres_mismatch: cpu_light forbids TresPerNode GPU evidence."
        )


def _require_candidate_resource_evidence_v3(
    receipt: Mapping[str, object],
    requirements: Mapping[str, object],
    resource_command_template: object,
) -> None:
    """Validate exact array-element resource capture and the frozen allocation."""

    resource_stdout = receipt.get("resource_stdout")
    if not isinstance(resource_stdout, str) or not resource_stdout.strip():
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_resource_invalid: scontrol resource evidence is empty."
        )
    slurm_value = receipt.get("slurm")
    if not isinstance(slurm_value, Mapping):
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_schema_invalid: slurm identity is missing."
        )
    slurm = candidate_runtime_slurm_identity(values=slurm_value)
    if set(slurm_value) != CANDIDATE_RESOURCE_SLURM_KEYS or dict(slurm_value) != slurm:
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_schema_invalid: v3 Slurm identity is not exact."
        )
    query_id = str(slurm["resource_query_job_id"])
    expected_command = [
        str(value).format(resource_query_job_id=query_id)
        for value in resource_command_template
    ]
    if receipt.get("resource_command") != expected_command:
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_resource_invalid: scontrol resource command drifted."
        )
    fields = _strict_scontrol_fields(resource_stdout)
    forbidden = [
        field
        for field in CANDIDATE_FORBIDDEN_LEGACY_SCHEDULER_FIELDS
        if field in fields
    ]
    if forbidden:
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_legacy_field_forbidden: v3 cannot consume "
            + ", ".join(forbidden)
            + "."
        )
    required_fields = {
        "JobId",
        "Partition",
        "NodeList",
        "CPUs/Task",
        "TRES",
        "MinMemoryNode",
        "OverSubscribe",
    }
    array_job_id = str(slurm.get("slurm_array_job_id") or "")
    if array_job_id:
        required_fields.update({"ArrayJobId", "ArrayTaskId"})
    missing = sorted(required_fields - set(fields))
    if missing:
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_resource_invalid: missing scontrol fields "
            + ", ".join(missing)
            + "."
        )
    if array_job_id:
        captured_query_id = f"{fields['ArrayJobId']}_{fields['ArrayTaskId']}"
        job_id_matches = fields["JobId"] in {
            query_id,
            str(slurm["slurm_job_id"]),
        }
    else:
        captured_query_id = fields["JobId"]
        job_id_matches = fields["JobId"] == query_id
        if "ArrayJobId" in fields or "ArrayTaskId" in fields:
            raise OperationalTwinProtocolError(
                "candidate_resource_receipt_resource_mismatch: non-array capture contains array identity fields."
            )
    if (
        captured_query_id != query_id
        or not job_id_matches
        or fields["Partition"] != str(requirements.get("partition"))
        or fields["NodeList"] != str(requirements.get("nodelist"))
        or fields["CPUs/Task"] != str(requirements.get("cpus_per_task"))
    ):
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_resource_mismatch: scontrol element identity, node, partition, or CPU evidence drifted."
        )
    tres = _strict_tres_fields(fields["TRES"], label="TRES")
    if tres.get("cpu") != str(requirements.get("cpus_per_task")):
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_resource_mismatch: TRES.cpu differs from the frozen CPU allocation."
        )
    memory = fields["MinMemoryNode"].upper()
    if not re.fullmatch(r"[1-9][0-9]*[KMGT]", memory):
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_resource_invalid: MinMemoryNode has a malformed unit."
        )
    if _memory_mebibytes(memory) != _memory_mebibytes(requirements.get("memory")):
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_resource_mismatch: MinMemoryNode differs from the frozen memory allocation."
        )
    if fields["OverSubscribe"] != "OK" or requirements.get("exclusive") is not False:
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_exclusivity_mismatch: OverSubscribe=OK is required for the frozen non-exclusive profile."
        )
    expected_gres = requirements.get("gres")
    if expected_gres == "gpu:1":
        if fields.get("TresPerNode") != "gres:gpu:1":
            raise OperationalTwinProtocolError(
                "candidate_resource_receipt_gres_mismatch: gpu_single requires exactly TresPerNode=gres:gpu:1."
            )
    elif "TresPerNode" in fields:
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_gres_mismatch: cpu_light forbids TresPerNode GPU evidence."
        )


def _require_candidate_resource_evidence(
    receipt: Mapping[str, object],
    requirements: Mapping[str, object],
    resource_command_template: object,
) -> None:
    """Validate normalized and raw scheduler evidence for the frozen resources."""
    allocation = receipt.get("resource_allocation")
    if (
        not isinstance(allocation, Mapping)
        or set(allocation) != CANDIDATE_RESOURCE_ALLOCATION_KEYS
        or dict(allocation) != dict(requirements)
    ):
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_allocation_mismatch: normalized CPU, memory, GRES, or exclusivity evidence drifted."
        )
    resource_stdout = receipt.get("resource_stdout")
    if not isinstance(resource_stdout, str) or not resource_stdout.strip():
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_resource_invalid: scontrol resource evidence is empty."
        )
    if receipt.get("resource_stdout_sha256") != sha256_text(resource_stdout):
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_resource_hash_mismatch: scontrol resource evidence hash drifted."
        )
    schema_version = receipt.get("schema_version")
    if schema_version == CANDIDATE_RESOURCE_RECEIPT_SCHEMA_VERSION_V1:
        _require_candidate_resource_evidence_v1(
            receipt,
            requirements,
            resource_command_template,
        )
        return
    if schema_version == CANDIDATE_RESOURCE_RECEIPT_SCHEMA_VERSION_V2:
        _require_candidate_resource_evidence_v2(
            receipt,
            requirements,
            resource_command_template,
        )
        return
    if schema_version in {CANDIDATE_RESOURCE_RECEIPT_SCHEMA_VERSION_V3,
                          "neuronsifter-operational-twin-candidate-resource-environment-receipt-v4"}:
        _require_candidate_resource_evidence_v3(
            receipt,
            requirements,
            resource_command_template,
        )
        return
    raise OperationalTwinProtocolError(
        "candidate_resource_receipt_schema_invalid: unsupported receipt schema version."
    )


def _require_live_candidate_resource_placement(
    requirements: Mapping[str, object],
    slurm: Mapping[str, object],
    environment: Mapping[str, object],
) -> None:
    """Match the live allocation to node, partition, CPU, memory, GRES, and GPU policy."""
    expected_node = str(requirements["nodelist"])
    expected_partition = str(requirements["partition"])
    if (
        slurm.get("slurm_job_partition") != expected_partition
        or slurm.get("slurm_job_nodelist") != expected_node
        or slurm.get("hostname") != expected_node
    ):
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_placement_mismatch: live node or partition differs from the frozen placement."
        )
    try:
        live_cpus = int(str(slurm.get("slurm_cpus_per_task")))
    except ValueError as error:
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_cpu_mismatch: live CPUs per task are missing or invalid."
        ) from error
    expected_cpus = int(requirements["cpus_per_task"])
    if live_cpus != expected_cpus:
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_cpu_mismatch: live CPUs per task differ from the frozen profile."
        )
    expected_memory = _memory_mebibytes(requirements["memory"])
    memory_per_node = _memory_mebibytes(slurm.get("slurm_mem_per_node"))
    memory_per_cpu = _memory_mebibytes(slurm.get("slurm_mem_per_cpu"))
    live_memory = memory_per_node
    if live_memory is None and memory_per_cpu is not None:
        live_memory = memory_per_cpu * live_cpus
    if live_memory != expected_memory:
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_memory_mismatch: live memory differs from the frozen profile."
        )
    def device_tokens(value: object) -> list[str]:
        text = str(value or "").strip()
        if text in {"", "(null)", "N/A", "None"}:
            return []
        tokens = text.split(",")
        if any(not token.strip() for token in tokens) or len(set(tokens)) != len(tokens):
            raise OperationalTwinProtocolError(
                "candidate_resource_receipt_gpu_mismatch: Slurm or CUDA device identity is malformed or duplicated."
            )
        return tokens

    job_gpu_tokens = device_tokens(slurm.get("slurm_job_gpus"))
    step_gpu_tokens = device_tokens(slurm.get("slurm_step_gpus"))
    visible_gpu_tokens = device_tokens(environment.get("cuda_visible_devices"))
    gpus_on_node_text = str(slurm.get("slurm_gpus_on_node") or "").strip()
    job_gres = str(slurm.get("slurm_job_gres") or "").strip()
    if requirements.get("gres") == "gpu:1":
        if job_gres and job_gres not in {"gpu:1", "gres/gpu:1"}:
            raise OperationalTwinProtocolError(
                "candidate_resource_receipt_gres_mismatch: live Slurm GRES differs from gpu:1."
            )
        if (
            len(job_gpu_tokens) != 1
            or gpus_on_node_text != "1"
            or (step_gpu_tokens and step_gpu_tokens != job_gpu_tokens)
            or environment.get("cuda_available") is not True
            or int(environment.get("cuda_device_count") or 0) != 1
            or len(visible_gpu_tokens) != 1
            or visible_gpu_tokens != job_gpu_tokens
        ):
            raise OperationalTwinProtocolError(
                "candidate_resource_receipt_gpu_mismatch: gpu_single requires one consistent Slurm-allocated and CUDA-visible GPU."
            )
    else:
        if (
            job_gres not in {"", "(null)", "N/A", "None"}
            or job_gpu_tokens
            or step_gpu_tokens
            or gpus_on_node_text not in {"", "0", "(null)", "N/A", "None"}
            or environment.get("cuda_available") is not False
            or int(environment.get("cuda_device_count") or 0) != 0
            or visible_gpu_tokens
        ):
            raise OperationalTwinProtocolError(
                "candidate_resource_receipt_cpu_gpu_mismatch: cpu_light forbids GRES, GPU allocation, CUDA, and visible devices."
            )


def validate_candidate_protocol_config(
    config: Mapping[str, object], root: Path
) -> list[dict[str, str]]:
    """Validate the prospective candidate-lineage config without touching sealed content."""
    blockers: list[dict[str, str]] = []
    try:
        protocol = _candidate_protocol(config)
    except OperationalTwinProtocolError as error:
        return [_block("missing_candidate_protocol_config", str(error))]
    schema_value = config.get(CANDIDATE_SCHEMA_KEY)
    if not isinstance(schema_value, Mapping):
        return [_block("missing_candidate_protocol_schema", f"{CANDIDATE_SCHEMA_KEY} is absent.")]
    schema = dict(schema_value)
    required_top = schema.get("required_top_level_fields")
    if not isinstance(required_top, list):
        blockers.append(_block("candidate_schema_invalid", "required_top_level_fields must be a list."))
        required_top = []
    if set(protocol) != set(str(value) for value in required_top):
        blockers.append(
            _block(
                "candidate_protocol_top_level_fields_mismatch",
                "Candidate protocol has missing or extra top-level fields.",
            )
        )
    if protocol.get("schema_version") != schema.get("protocol_schema_version"):
        blockers.append(_block("candidate_protocol_schema_version_mismatch", "Candidate protocol schema version drifted."))
    if protocol.get("authoritative_date") != "2026-09-08":
        blockers.append(_block("candidate_authoritative_date_mismatch", "Candidate authoritative date must be 2026-09-08."))
    if protocol.get("evidence_class") != "controlled_world_non_empirical":
        blockers.append(_block("candidate_evidence_class_mismatch", "Candidate evidence class drifted."))
    if protocol.get("parent_candidate_lineage_uid") is not None:
        blockers.append(_block("candidate_parent_lineage_not_null", "New candidate lineage must have a null parent."))

    scheduler_contract_value = protocol.get("scheduler_evidence_contract")
    scheduler_contract_is_exact = _validate_candidate_exact_section(
        schema,
        "scheduler_evidence_contract",
        scheduler_contract_value,
        blockers=blockers,
        error_code="candidate_scheduler_evidence_contract_drift",
    )
    if scheduler_contract_is_exact:
        scheduler_contract = dict(scheduler_contract_value)
        if (
            scheduler_contract.get("contract_version")
            != CANDIDATE_SCHEDULER_EVIDENCE_CONTRACT_VERSION
            or scheduler_contract.get("field_mapping")
            != CANDIDATE_SCHEDULER_FIELD_MAPPING
            or scheduler_contract.get("forbidden_legacy_fields")
            != CANDIDATE_FORBIDDEN_LEGACY_SCHEDULER_FIELDS
        ):
            blockers.append(
                _block(
                    "candidate_scheduler_evidence_contract_drift",
                    "Scheduler evidence version, field mapping, or legacy-field prohibition drifted.",
                )
            )
        for fixture in dict(scheduler_contract.get("retained_fixtures", {})).values():
            if not isinstance(fixture, Mapping):
                blockers.append(
                    _block(
                        "candidate_scheduler_fixture_invalid",
                        "Retained scheduler fixture binding must be a mapping.",
                    )
                )
                continue
            try:
                fixture_payload = read_repo_json(root, str(fixture["path"]))
            except (KeyError, OperationalTwinProtocolError, json.JSONDecodeError) as error:
                blockers.append(
                    _block("candidate_scheduler_fixture_unavailable", str(error))
                )
                continue
            if str(fixture_payload.get("source_job_id")) != str(
                fixture.get("source_job_id")
            ):
                blockers.append(
                    _block(
                        "candidate_scheduler_fixture_identity_mismatch",
                        "Retained scheduler fixture job identity drifted.",
                    )
                )

    initialization = _require_exact_keys(
        protocol.get("initialization"),
        {
            "initialization_mode",
            "resume_allowed",
            "resume_input",
            "history_input",
            "checkpoint_input",
            "optimizer_state_input",
            "model_state_input",
            "calibration_state_input",
            "historical_state_use_allowed",
        },
        label="initialization",
        blockers=blockers,
    )
    if initialization.get("initialization_mode") != "fresh_random":
        blockers.append(_block("candidate_initialization_not_fresh", "Candidate initialization must be fresh_random."))
    if initialization.get("resume_allowed") is not False or initialization.get("historical_state_use_allowed") is not False:
        blockers.append(_block("candidate_state_reuse_open", "Resume and historical state reuse must be false."))
    for field in CANDIDATE_FORBIDDEN_STATE_INPUT_FIELDS:
        if initialization.get(field) is not None:
            blockers.append(_block("candidate_forbidden_state_input", f"{field} must be explicit null."))

    try:
        expected_scientific = _candidate_scientific_contract_from_v1(config)
    except OperationalTwinProtocolError as error:
        blockers.append(_block("candidate_v1_authority_missing", str(error)))
        expected_scientific = {}
    scientific = protocol.get("scientific_contract")
    if scientific != expected_scientific:
        blockers.append(
            _block(
                "candidate_scientific_contract_drift",
                "Candidate methods, seeds, data, runtime, objectives, metrics, statistics, selection, calibration, or failure semantics differ from v1.",
            )
        )
    if isinstance(scientific, Mapping):
        methods = [str(dict(row).get("method_id", "")) for row in scientific.get("methods", [])]
        seeds = [int(seed) for seed in scientific.get("seeds", [])]
        if methods != REQUIRED_METHOD_IDS:
            blockers.append(_block("candidate_method_identity_mismatch", "Candidate must preserve the exact five methods in order."))
        if seeds != [360101, 360211, 360307]:
            blockers.append(_block("candidate_seed_identity_mismatch", "Candidate must preserve seeds 360101, 360211, and 360307."))

    if protocol.get("candidate_lineage_uid") != compute_candidate_lineage_uid(protocol):
        blockers.append(_block("candidate_lineage_uid_mismatch", "candidate_lineage_uid does not recompute from its frozen basis."))
    if protocol.get("protocol_uid") != compute_candidate_protocol_uid(protocol):
        blockers.append(_block("candidate_protocol_uid_mismatch", "protocol_uid does not recompute from the complete frozen basis."))

    schema_identity_fields = schema.get("required_identity_fields")
    if schema_identity_fields != list(CANDIDATE_IDENTITY_FIELDS):
        blockers.append(
            _block(
                "candidate_schema_invalid",
                "Schema required_identity_fields differ from the runtime identity contract.",
            )
        )
    _validate_candidate_exact_section(
        schema,
        "identity_policy",
        protocol.get("identity_policy"),
        blockers=blockers,
        error_code="candidate_identity_policy_drift",
        field_error_codes={
            "required_identity_fields": "candidate_identity_field_contract_mismatch",
            "manifest_provided_attempt_uid_required": "candidate_attempt_manifest_binding_open",
            "directory_derived_attempt_uid_forbidden": "candidate_directory_attempt_derivation_open",
        },
    )

    execution_authority = _require_exact_keys(
        protocol.get("execution_authority_contract"),
        {
            "authority_schema_version",
            "resource_receipt_schema_version",
            "authority_role",
            "positive_authority_terminal",
            "positive_resource_receipt_terminal",
            "approved_protocol_review_path",
            "required_review_schema_version",
            "required_review_round_id",
            "required_review_direction_id",
            "required_review_decision",
            "scheduler_evidence_contract_version",
            "candidate_execution_authority_path_template",
            "resource_environment_receipt_path_template",
            "fresh_sinfo_command",
            "fresh_resource_command",
            "maximum_resource_receipt_age_seconds",
            "independent_execution_authority_required_before_dispatch",
            "resource_environment_receipt_required_before_dispatch",
            "approved_review_binding_required",
            "source_commit_binding_required",
            "source_identity_binding_required",
            "assignment_binding_required",
            "attempt_binding_required",
            "exact_placement_required",
            "cpu_memory_gres_exclusivity_evidence_required",
            "terminal_provenance_binding_required",
            "authority_bound_in_this_round",
            "resource_receipt_bound_in_this_round",
        },
        label="execution_authority_contract",
        blockers=blockers,
    )
    expected_execution_authority = {
        "authority_schema_version": CANDIDATE_EXECUTION_AUTHORITY_SCHEMA_VERSION_V3,
        "resource_receipt_schema_version": CANDIDATE_RESOURCE_RECEIPT_SCHEMA_VERSION_V3,
        "authority_role": "operational_twin_candidate_execution",
        "positive_authority_terminal": CANDIDATE_EXECUTION_AUTHORITY_TERMINAL,
        "positive_resource_receipt_terminal": CANDIDATE_RESOURCE_RECEIPT_TERMINAL,
        "approved_protocol_review_path": "orchestrator/rounds/round-47-repair-operational-twin-candidate-execution-interfaces/review-record.json",
        "required_review_schema_version": "review-record-v4",
        "required_review_round_id": "round-47-repair-operational-twin-candidate-execution-interfaces",
        "required_review_direction_id": "repair-operational-twin-candidate-execution-interfaces",
        "required_review_decision": "approved",
        "scheduler_evidence_contract_version": CANDIDATE_SCHEDULER_EVIDENCE_CONTRACT_VERSION,
        "candidate_execution_authority_path_template": "Output/audits/operational_twin/candidate_execution_authorities/{candidate_lineage_uid}/{assignment_uid}/{attempt_uid}.json",
        "resource_environment_receipt_path_template": "Output/audits/operational_twin/candidate_resource_environment_receipts/{candidate_lineage_uid}/{assignment_uid}/{attempt_uid}.json",
        "fresh_sinfo_command": [
            "sinfo",
            "--noheader",
            "--nodes={nodelist}",
            "--format=%N|%P|%T|%G",
        ],
        "fresh_resource_command": [
            "scontrol",
            "--oneliner",
            "show",
            "job",
            "{resource_query_job_id}",
        ],
        "maximum_resource_receipt_age_seconds": 1800,
        "independent_execution_authority_required_before_dispatch": True,
        "resource_environment_receipt_required_before_dispatch": True,
        "approved_review_binding_required": True,
        "source_commit_binding_required": True,
        "source_identity_binding_required": True,
        "assignment_binding_required": True,
        "attempt_binding_required": True,
        "exact_placement_required": True,
        "cpu_memory_gres_exclusivity_evidence_required": True,
        "terminal_provenance_binding_required": True,
        "authority_bound_in_this_round": False,
        "resource_receipt_bound_in_this_round": False,
    }
    schema_execution_authority = _require_exact_keys(
        schema.get("execution_authority_contract"),
        set(expected_execution_authority),
        label="schema.execution_authority_contract",
        blockers=blockers,
    )
    if schema_execution_authority != expected_execution_authority:
        blockers.append(
            _block(
                "candidate_execution_authority_schema_drift",
                "The candidate schema execution-authority contract differs from the runtime contract.",
            )
        )
    if execution_authority != expected_execution_authority:
        blockers.append(
            _block(
                "candidate_execution_authority_contract_drift",
                "The later-round execution-authority/resource-receipt contract drifted.",
            )
        )

    _validate_candidate_exact_section(
        schema,
        "execution_contract",
        protocol.get("execution_contract"),
        blockers=blockers,
        error_code="candidate_execution_contract_drift",
        field_error_codes={
            "attempt_slots": "candidate_attempt_slots_invalid",
            "profiles": "candidate_resource_profile_drift",
            "wall_time_by_method": "candidate_wall_time_invalid",
        },
    )

    output_paths = dict(protocol.get("output_paths", {}))
    expected_paths = {
        "protocol_root": "Output/results/operational_twin/candidate_lineage_protocol/execution_interface_guarded",
        "candidate_lineage_manifest": "Output/results/operational_twin/candidate_lineage_protocol/execution_interface_guarded/candidate_lineage_manifest.json",
        "assignment_manifest": "Output/results/operational_twin/candidate_lineage_protocol/execution_interface_guarded/assignment_manifest.json",
        "assignment_commands": "Output/results/operational_twin/candidate_lineage_protocol/execution_interface_guarded/assignment_commands.json",
        "preflight_authority": "Output/audits/operational_twin_candidate_lineage_execution_interface_guarded_preflight.json",
        "preflight_report": "Output/reports/operational_twin_candidate_lineage_execution_interface_guarded_preflight.md",
        "interface_repair_audit": "Output/audits/operational_twin_candidate_execution_interfaces.json",
        "interface_repair_report": "Output/reports/operational_twin_candidate_execution_interfaces.md",
        "attempt_root_template": "Output/results/operational_twin/candidate_lineages/{candidate_lineage_uid}/assignments/{assignment_uid}/attempts/{attempt_uid}",
        "candidate_calibration_execution_authority": "Output/audits/operational_twin_candidate_lineage_calibration_execution_authority.json",
    }
    if output_paths != expected_paths:
        blockers.append(_block("candidate_output_contract_drift", "Candidate output paths differ from the semantic frozen contract."))
    for label, value in output_paths.items():
        if label in {
            "candidate_calibration_execution_authority",
            "interface_repair_audit",
            "interface_repair_report",
        }:
            continue
        try:
            declared = str(value).format(
                candidate_lineage_uid="candidate",
                assignment_uid="assignment",
                attempt_uid="attempt",
            )
            if label == "attempt_root_template":
                candidate_attempt_path(
                    root,
                    declared,
                    require_absent=False,
                )
            else:
                resolved = repo_path(root, declared)
                resolved.relative_to(root.resolve())
        except (ValueError, KeyError) as error:
            blockers.append(_block("candidate_output_path_invalid", f"{label}: {error}"))

    for key in REQUIRED_ZERO_ACCESS_COUNTERS:
        if dict(protocol.get("access_counters", {})).get(key) != 0:
            blockers.append(_block("candidate_nonzero_access_counter", f"{key} must remain zero."))
    if set(dict(protocol.get("access_counters", {}))) != set(REQUIRED_ZERO_ACCESS_COUNTERS):
        blockers.append(_block("candidate_access_counter_fields_mismatch", "Candidate access counter fields drifted."))
    required_closed_flags = schema.get("required_closed_flags")
    if not isinstance(required_closed_flags, list) or any(
        not isinstance(field, str) for field in required_closed_flags
    ):
        blockers.append(
            _block(
                "candidate_schema_invalid",
                "required_closed_flags must be a list of field names.",
            )
        )
        required_closed_flags = []
    closed_schema = schema.get("closed_flags")
    if (
        isinstance(closed_schema, Mapping)
        and closed_schema.get("required") != required_closed_flags
    ):
        blockers.append(
            _block(
                "candidate_schema_invalid",
                "closed_flags.required must equal required_closed_flags.",
            )
        )
    _require_exact_keys(
        protocol.get("closed_flags"),
        set(required_closed_flags),
        label="closed_flags",
        blockers=blockers,
    )
    _validate_candidate_exact_section(
        schema,
        "closed_flags",
        protocol.get("closed_flags"),
        blockers=blockers,
        error_code="candidate_closed_flags_drift",
        field_error_codes={field: "candidate_downstream_gate_open" for field in required_closed_flags},
    )
    terminal_value = protocol.get("terminal_policy")
    _validate_candidate_exact_section(
        schema,
        "terminal_policy",
        terminal_value,
        blockers=blockers,
        error_code="candidate_terminal_policy_drift",
        field_error_codes={
            "training_executed": "candidate_execution_evidence_present",
            "calibration_fitting_executed": "candidate_execution_evidence_present",
            "checkpoint_created": "candidate_execution_evidence_present",
        },
    )

    historical_value = protocol.get("historical_boundary")
    historical = (
        dict(historical_value) if isinstance(historical_value, Mapping) else {}
    )
    historical_is_exact = _validate_candidate_exact_section(
        schema,
        "historical_boundary",
        historical_value,
        blockers=blockers,
        error_code="candidate_historical_boundary_drift",
    )
    if historical_is_exact:
        try:
            audit = read_repo_json(root, str(historical["audit_path"]))
            boundary = dict(audit.get("boundary", {}))
            if audit.get("terminal") != historical["candidate_status"]:
                blockers.append(_block("historical_terminal_drift", "Round 38 terminal is not evidence-invalid."))
            if boundary.get("sufficient_training") is not False or boundary.get("twin_qualification_pass") is not False or boundary.get("formal_qualification") is not False:
                blockers.append(_block("historical_boundary_open", "Round 38 scientific gates are not closed."))
        except (OperationalTwinProtocolError, json.JSONDecodeError) as error:
            blockers.append(_block("historical_authority_unavailable", str(error)))

    sealed_value = protocol.get("sealed_manifest_metadata")
    sealed = dict(sealed_value) if isinstance(sealed_value, Mapping) else {}
    sealed_is_exact = _validate_candidate_exact_section(
        schema,
        "sealed_manifest_metadata",
        sealed_value,
        blockers=blockers,
        error_code="sealed_metadata_identity_drift",
        field_error_codes={
            "metadata_source": "sealed_metadata_boundary_invalid",
            "first_use_policy": "sealed_metadata_boundary_invalid",
            "sealed_content_opened": "sealed_metadata_boundary_invalid",
        },
    )
    if sealed_is_exact:
        try:
            metadata_payload = yaml.safe_load(
                read_repo_bytes(root, str(sealed["metadata_source"]))
            ) or {}
            historical_provenance = dict(
                dict(metadata_payload.get("controlled_world_open_action_identification", {})).get(
                    "historical_benchmark_provenance", {}
                )
            )
            for field in (
                "complete_test_manifest_git_blob",
                "freeze_uid",
                "test_manifest_uid",
                "test_content_identity",
            ):
                if sealed.get(field) != historical_provenance.get(field):
                    blockers.append(_block("sealed_metadata_identity_drift", f"sealed_manifest_metadata.{field} drifted."))
        except (OperationalTwinProtocolError, yaml.YAMLError) as error:
            blockers.append(_block("sealed_metadata_authority_unavailable", str(error)))

    supersession_value = protocol.get("supersession_contract")
    supersession_is_exact = _validate_candidate_exact_section(
        schema,
        "supersession_contract",
        supersession_value,
        blockers=blockers,
        error_code="candidate_supersession_contract_drift",
    )
    if supersession_value != CANDIDATE_SUPERSESSION_CONTRACT:
        blockers.append(
            _block(
                "candidate_supersession_contract_drift",
                "The successor freeze must bind the exact rejected predecessor contract.",
            )
        )
        supersession_is_exact = False
    if supersession_is_exact:
        supersession = dict(supersession_value)
        review = dict(supersession["rejected_review_record"])
        if (root / ".git").exists():
            try:
                review_bytes = _git_object_bytes(
                    root,
                    str(review["git_object_id"]),
                )
                if hashlib.sha256(review_bytes).hexdigest() != review["sha256"]:
                    blockers.append(
                        _block(
                            "candidate_supersession_review_drift",
                            "The rejected Round 43 review record is not byte-identical to the bound predecessor review.",
                        )
                    )
                review_payload = json.loads(review_bytes.decode("utf-8"))
                for field in ("schema_version", "round_id", "direction_id", "decision"):
                    if review_payload.get(field) != review[field]:
                        blockers.append(
                            _block(
                                "candidate_supersession_review_drift",
                                f"Rejected review field {field} drifted from the supersession contract.",
                            )
                        )
            except (OperationalTwinProtocolError, json.JSONDecodeError) as error:
                blockers.append(_block("candidate_supersession_review_unavailable", str(error)))

        predecessor_paths: set[str] = set()
        for role, identity_value in dict(supersession["predecessor_artifacts"]).items():
            identity = dict(identity_value)
            predecessor_path = str(identity["path"])
            predecessor_paths.add(predecessor_path)
            try:
                content = read_repo_bytes(root, predecessor_path)
                if hashlib.sha256(content).hexdigest() != identity["sha256"]:
                    blockers.append(
                        _block(
                            "candidate_supersession_artifact_drift",
                            f"Rejected predecessor artifact {role} is not byte-identical.",
                        )
                    )
                if (root / ".git").exists():
                    committed_content = _git_commit_path_bytes(
                        root,
                        str(supersession["predecessor_implementation_commit"]),
                        predecessor_path,
                    )
                    if hashlib.sha256(committed_content).hexdigest() != identity["sha256"]:
                        blockers.append(
                            _block(
                                "candidate_supersession_commit_artifact_drift",
                                f"Rejected predecessor artifact {role} does not match the bound implementation commit.",
                            )
                        )
                if role != "preflight_report":
                    predecessor_payload = json.loads(content.decode("utf-8"))
                    if (
                        predecessor_payload.get("candidate_lineage_uid")
                        != supersession["predecessor_candidate_lineage_uid"]
                        or predecessor_payload.get("protocol_uid")
                        != supersession["predecessor_protocol_uid"]
                    ):
                        blockers.append(
                            _block(
                                "candidate_supersession_artifact_identity_drift",
                                f"Rejected predecessor artifact {role} carries a different lineage or protocol UID.",
                            )
                        )
            except (OperationalTwinProtocolError, json.JSONDecodeError) as error:
                blockers.append(
                    _block(
                        "candidate_supersession_artifact_unavailable",
                        f"{role}: {error}",
                    )
                )
        amendment = dict(supersession["resource_receipt_amendment_basis"])
        for evidence_role in (
            "approved_round_43_review_record",
            "approved_round_44_review_record",
            "approved_rev_007_update",
            "approved_rev_007_update_review",
        ):
            identity = dict(amendment[evidence_role])
            try:
                content = read_repo_bytes(root, str(identity["path"]))
                if hashlib.sha256(content).hexdigest() != identity["sha256"]:
                    blockers.append(
                        _block(
                            "candidate_amendment_evidence_drift",
                            f"Amendment evidence {evidence_role} is not byte-identical.",
                        )
                    )
                if (
                    (root / ".git").exists()
                    and _git_object_bytes(root, str(identity["git_object_id"]))
                    != content
                ):
                    blockers.append(
                        _block(
                            "candidate_amendment_evidence_git_drift",
                            f"Amendment evidence {evidence_role} differs from its Git object.",
                        )
                    )
                if evidence_role.endswith("review_record"):
                    payload = json.loads(content.decode("utf-8"))
                    for field in (
                        "schema_version",
                        "round_id",
                        "direction_id",
                        "decision",
                    ):
                        if payload.get(field) != identity[field]:
                            blockers.append(
                                _block(
                                    "candidate_amendment_review_identity_drift",
                                    f"Amendment review {evidence_role}.{field} drifted.",
                                )
                            )
                elif evidence_role == "approved_rev_007_update_review" and (
                    "### Decision\n\nAPPROVED" not in content.decode("utf-8")
                    or identity.get("decision") != "APPROVED"
                ):
                    blockers.append(
                        _block(
                            "candidate_amendment_review_identity_drift",
                            "The rev-007 roadmap-update review is not approved.",
                        )
                    )
            except (OperationalTwinProtocolError, json.JSONDecodeError) as error:
                blockers.append(
                    _block(
                        "candidate_amendment_evidence_unavailable",
                        f"{evidence_role}: {error}",
                    )
                )

        repair = dict(supersession["execution_interface_repair_basis"])
        if (root / ".git").exists():
            for commit_field in ("baseline_commit", "merged_round_46_commit"):
                completed = subprocess.run(
                    [
                        "git",
                        "merge-base",
                        "--is-ancestor",
                        str(repair[commit_field]),
                        "HEAD",
                    ],
                    cwd=root,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if completed.returncode != 0:
                    blockers.append(
                        _block(
                            "candidate_interface_repair_commit_unavailable",
                            f"{commit_field} is not an ancestor of the current source.",
                        )
                    )
        for evidence_role in (
            "approved_round_46_review_record",
            "implementation_notes",
            "blocker_summary",
        ):
            identity = dict(repair[evidence_role])
            try:
                content = read_repo_bytes(root, str(identity["path"]))
                if hashlib.sha256(content).hexdigest() != identity["sha256"]:
                    blockers.append(
                        _block(
                            "candidate_interface_repair_evidence_drift",
                            f"Round 46 repair evidence {evidence_role} is not byte-identical.",
                        )
                    )
                if (
                    (root / ".git").exists()
                    and _git_object_bytes(root, str(identity["git_object_id"]))
                    != content
                ):
                    blockers.append(
                        _block(
                            "candidate_interface_repair_evidence_git_drift",
                            f"Round 46 repair evidence {evidence_role} differs from its Git object.",
                        )
                    )
                if (root / ".git").exists():
                    merged_content = _git_commit_path_bytes(
                        root,
                        str(repair["merged_round_46_commit"]),
                        str(identity["path"]),
                    )
                    if merged_content != content:
                        blockers.append(
                            _block(
                                "candidate_interface_repair_merge_drift",
                                f"Round 46 merged evidence {evidence_role} drifted.",
                            )
                        )
                if evidence_role == "approved_round_46_review_record":
                    payload = json.loads(content.decode("utf-8"))
                    for field in (
                        "schema_version",
                        "round_id",
                        "direction_id",
                        "decision",
                    ):
                        if payload.get(field) != identity[field]:
                            blockers.append(
                                _block(
                                    "candidate_interface_repair_review_drift",
                                    f"Round 46 review {field} drifted.",
                                )
                            )
            except (OperationalTwinProtocolError, json.JSONDecodeError) as error:
                blockers.append(
                    _block(
                        "candidate_interface_repair_evidence_unavailable",
                        f"{evidence_role}: {error}",
                    )
                )
        expected_boundary = {
            "terminal_decision": "OPERATIONAL_TWIN_CANDIDATE_LINEAGE_EXECUTION_INCOMPLETE",
            "initial_attempt_count": 15,
            "failed_initial_attempt_count": 15,
            "materialized_attempt_count": 10,
            "prepublication_failure_count": 5,
            "evaluation_summary_count": 0,
            "recovery_attempt_count": 0,
            "scheduler_receipt_count": 0,
            "candidate_execution_catalog_created": False,
            "sufficiency_adjudicated": False,
            "scientific_twin_result": "NOT_RUN",
            "access_counters": {key: 0 for key in REQUIRED_ZERO_ACCESS_COUNTERS},
        }
        if repair.get("operational_boundary") != expected_boundary:
            blockers.append(
                _block(
                    "candidate_interface_repair_boundary_drift",
                    "The immutable Round 46 incomplete-execution boundary drifted.",
                )
            )

        generation_paths: set[str] = set()
        generations = dict(amendment["predecessor_generations"])
        for generation_name, generation_value in generations.items():
            generation = dict(generation_value)
            generation_protocol_uid = str(generation["protocol_uid"])
            artifacts = dict(generation["artifacts"])
            if set(artifacts) != {
                "candidate_lineage_manifest",
                "assignment_manifest",
                "assignment_commands",
                "preflight_authority",
                "preflight_report",
            }:
                blockers.append(
                    _block(
                        "candidate_predecessor_generation_incomplete",
                        f"Predecessor generation {generation_name} does not contain five artifacts.",
                    )
                )
            for role, identity_value in artifacts.items():
                identity = dict(identity_value)
                predecessor_path = str(identity["path"])
                generation_paths.add(predecessor_path)
                try:
                    content = read_repo_bytes(root, predecessor_path)
                    if hashlib.sha256(content).hexdigest() != identity["sha256"]:
                        blockers.append(
                            _block(
                                "candidate_predecessor_generation_drift",
                                f"{generation_name}/{role} is not byte-identical.",
                            )
                        )
                    if role != "preflight_report":
                        payload = json.loads(content.decode("utf-8"))
                        if (
                            payload.get("candidate_lineage_uid")
                            != protocol["candidate_lineage_uid"]
                            or payload.get("protocol_uid")
                            != generation_protocol_uid
                        ):
                            blockers.append(
                                _block(
                                    "candidate_predecessor_generation_identity_drift",
                                    f"{generation_name}/{role} carries a different lineage or protocol UID.",
                                )
                            )
                except (OperationalTwinProtocolError, json.JSONDecodeError) as error:
                    blockers.append(
                        _block(
                            "candidate_predecessor_generation_unavailable",
                            f"{generation_name}/{role}: {error}",
                        )
                    )
        if len(generation_paths) != 20:
            blockers.append(
                _block(
                    "candidate_predecessor_generation_count_mismatch",
                    "Exactly 20 byte-distinct predecessor artifact paths are required.",
                )
            )
        predecessor_paths.update(generation_paths)
        successor_paths = {
            str(output_paths[key])
            for key in (
                "candidate_lineage_manifest",
                "assignment_manifest",
                "assignment_commands",
                "preflight_authority",
                "preflight_report",
            )
            if key in output_paths
        }
        if predecessor_paths & successor_paths:
            blockers.append(
                _block(
                    "candidate_supersession_output_collision",
                    "Successor protocol artifacts must use paths disjoint from all rejected predecessor artifacts.",
                )
            )

    source_bindings_value = protocol.get("source_bindings")
    source_bindings = (
        dict(source_bindings_value)
        if isinstance(source_bindings_value, Mapping)
        else {}
    )
    source_bindings_are_exact = _validate_candidate_exact_section(
        schema,
        "source_bindings",
        source_bindings_value,
        blockers=blockers,
        error_code="candidate_source_bindings_drift",
        field_error_codes={
            "snapshot_repair_prospective_only": "round41_scope_drift",
        },
    )
    if source_bindings_are_exact:
        trainer_path = str(source_bindings["trainer"])
        test_path = str(source_bindings["protocol_regression_test"])
        try:
            trainer_text = read_repo_bytes(root, trainer_path).decode("utf-8")
            test_text = read_repo_bytes(root, test_path).decode("utf-8")
            if "best_payload = deepcopy(payload)" not in trainer_text:
                blockers.append(_block("round41_snapshot_source_missing", "Trainer lacks complete-payload deepcopy selection."))
            if "test_selected_checkpoint_snapshot_survives_later_training_and_calibration" not in test_text:
                blockers.append(_block("round41_snapshot_regression_missing", "Round 41 production regression is absent."))
        except OperationalTwinProtocolError as error:
            blockers.append(_block("round41_source_unavailable", str(error)))
    return blockers


def build_candidate_assignments(protocol: Mapping[str, object]) -> list[dict[str, object]]:
    scientific = dict(protocol["scientific_contract"])
    execution = dict(protocol["execution_contract"])
    output_template = str(dict(protocol["output_paths"])["attempt_root_template"])
    assignments: list[dict[str, object]] = []
    for method in scientific["methods"]:
        method_row = dict(method)
        method_id = str(method_row["method_id"])
        resource_profile = "cpu_light" if method_id == "persistence_no_action" else "gpu_single"
        for seed_value in scientific["seeds"]:
            seed = int(seed_value)
            assignment_uid = compute_candidate_assignment_uid(protocol, method_id, seed)
            assignment_identity = {
                "candidate_lineage_uid": protocol["candidate_lineage_uid"],
                "protocol_uid": protocol["protocol_uid"],
                "assignment_uid": assignment_uid,
            }
            attempts: list[dict[str, object]] = []
            initial_uid = ""
            for slot in execution["attempt_slots"]:
                slot_row = dict(slot)
                attempt_uid = compute_candidate_attempt_uid(protocol, assignment_uid, str(slot_row["attempt_kind"]))
                if slot_row["attempt_kind"] == "initial":
                    initial_uid = attempt_uid
                output_root = output_template.format(
                    candidate_lineage_uid=protocol["candidate_lineage_uid"],
                    assignment_uid=assignment_uid,
                    attempt_uid=attempt_uid,
                )
                attempt = {
                    **candidate_identity_payload(
                        assignment_identity,
                        {"attempt_uid": attempt_uid},
                    ),
                    "attempt_kind": slot_row["attempt_kind"],
                    "attempt_ordinal": int(slot_row["attempt_ordinal"]),
                    "conditional": bool(slot_row["conditional"]),
                    "eligible_after_failure_class": slot_row["eligible_after_failure_class"],
                    "predecessor_attempt_uid": initial_uid if bool(slot_row["conditional"]) else None,
                    "output_root": output_root,
                    "immutable_output_root": True,
                    "fresh_random_initialization": True,
                    "state_resume_allowed": False,
                }
                attempt.update(
                    candidate_execution_artifact_paths(
                        protocol,
                        assignment_identity,
                        attempt,
                    )
                )
                attempts.append(attempt)
            assignments.append(
                {
                    "candidate_lineage_uid": protocol["candidate_lineage_uid"],
                    "protocol_uid": protocol["protocol_uid"],
                    "assignment_uid": assignment_uid,
                    "method_id": method_id,
                    "seed": seed,
                    "execution_contract": method_row["execution_contract"],
                    "runtime_family": method_row["runtime_family"],
                    "resource_profile": resource_profile,
                    "resource_requirements": dict(execution["profiles"])[resource_profile],
                    "wall_time": dict(execution["wall_time_by_method"])[method_id],
                    "capacity": method_row["capacity"],
                    "budget": method_row["budget"],
                    "metrics_contract": method_row["metrics_contract"],
                    "checkpoint_creation_allowed": method_row["checkpoint_creation_allowed"],
                    "calibration_eligible": method_row["calibration_eligible"],
                    "attempts": attempts,
                }
            )
    return assignments


def build_candidate_lineage_manifest(config: Mapping[str, object]) -> dict[str, object]:
    protocol = _candidate_protocol(config)
    assignments = build_candidate_assignments(protocol)
    payload: dict[str, object] = {
        "schema_version": CANDIDATE_MANIFEST_SCHEMA_VERSION,
        "status": "FROZEN",
        "candidate_lineage_uid": protocol["candidate_lineage_uid"],
        "protocol_uid": protocol["protocol_uid"],
        "preflight_uid": compute_candidate_preflight_uid(protocol),
        "parent_candidate_lineage_uid": None,
        "initialization": candidate_fresh_initialization_receipt(protocol),
        "scientific_contract_hash": canonical_hash(protocol["scientific_contract"]),
        "assignment_uids": [row["assignment_uid"] for row in assignments],
        "attempt_uids": [attempt["attempt_uid"] for row in assignments for attempt in row["attempts"]],
        "required_identity_fields": list(CANDIDATE_IDENTITY_FIELDS),
        "historical_boundary": protocol["historical_boundary"],
        "sealed_manifest_metadata": protocol["sealed_manifest_metadata"],
        "scheduler_evidence_contract": protocol["scheduler_evidence_contract"],
        "supersession_contract": protocol["supersession_contract"],
        "supersession_contract_hash": candidate_supersession_contract_hash(protocol),
        "attempt_root_template": dict(protocol["output_paths"])["attempt_root_template"],
        "execution_authority_contract": protocol["execution_authority_contract"],
        "access_counters": protocol["access_counters"],
        "closed_flags": protocol["closed_flags"],
        "training_executed": False,
        "calibration_fitting_executed": False,
        "checkpoint_created": False,
    }
    payload["candidate_manifest_hash"] = canonical_hash(payload)
    return payload


def build_candidate_assignment_manifest(
    config: Mapping[str, object], candidate_manifest: Mapping[str, object]
) -> dict[str, object]:
    protocol = _candidate_protocol(config)
    assignments = build_candidate_assignments(protocol)
    payload: dict[str, object] = {
        "schema_version": "neuronsifter-operational-twin-candidate-assignment-manifest-v1",
        "candidate_lineage_uid": protocol["candidate_lineage_uid"],
        "protocol_uid": protocol["protocol_uid"],
        "preflight_uid": compute_candidate_preflight_uid(protocol),
        "candidate_manifest_hash": candidate_manifest["candidate_manifest_hash"],
        "assignment_count": len(assignments),
        "attempt_slot_count": sum(len(row["attempts"]) for row in assignments),
        "method_ids": REQUIRED_METHOD_IDS,
        "seeds": [int(seed) for seed in dict(protocol["scientific_contract"])["seeds"]],
        "assignments": assignments,
        "initialization": candidate_fresh_initialization_receipt(protocol),
        "execution_authority_contract": protocol["execution_authority_contract"],
        "access_counters": protocol["access_counters"],
    }
    payload["assignment_manifest_hash"] = canonical_hash(payload)
    return payload


def command_catalog_hash(catalog: Mapping[str, object]) -> str:
    normalized = deepcopy(dict(catalog))
    normalized.pop("command_catalog_hash", None)
    for row in normalized.get("commands", []):
        argv = list(dict(row).get("argv", []))
        if "--command-catalog-hash" in argv:
            index = argv.index("--command-catalog-hash")
            if index + 1 < len(argv):
                argv[index + 1] = "__COMMAND_CATALOG_HASH__"
        dict(row)["argv"] = argv
        row["argv"] = argv
    return canonical_hash(normalized)


def expected_candidate_command_argv(
    config: Mapping[str, object],
    candidate_manifest: Mapping[str, object],
    assignment_manifest: Mapping[str, object],
    assignment: Mapping[str, object],
    attempt: Mapping[str, object],
    *,
    command_catalog_hash_value: object,
) -> list[str]:
    """Build the canonical exact argv sequence admitted for a candidate attempt."""
    protocol = _candidate_protocol(config)
    scientific = dict(protocol["scientific_contract"])
    paths = dict(protocol["output_paths"])
    return [
        "python",
        "Libs/exp/training/train_operational_twin_devcal.py",
        "--protocol-scope",
        "candidate-lineage",
        "--config",
        "Libs/configs/model/operational_twin.yaml",
        "--training-manifest",
        str(dict(scientific["manifests"])["training"]),
        "--development-manifest",
        str(dict(scientific["manifests"])["development"]),
        "--calibration-manifest",
        str(dict(scientific["manifests"])["calibration"]),
        "--package-receipt",
        "Input/processed/controlled_neural_world/open_action_identification/package_receipt.json",
        "--open-data-qualification",
        str(scientific["open_data_authority"]),
        "--calibration-execution-authority",
        str(paths["candidate_calibration_execution_authority"]),
        "--candidate-execution-authority",
        str(attempt["execution_authority"]),
        "--candidate-resource-environment-receipt",
        str(attempt["resource_environment_receipt"]),
        "--output-root",
        str(attempt["output_root"]),
        "--candidate-lineage-uid",
        str(assignment["candidate_lineage_uid"]),
        "--protocol-uid",
        str(assignment["protocol_uid"]),
        "--assignment-uid",
        str(assignment["assignment_uid"]),
        "--attempt-uid",
        str(attempt["attempt_uid"]),
        "--candidate-manifest-hash",
        str(candidate_manifest["candidate_manifest_hash"]),
        "--assignment-manifest-hash",
        str(assignment_manifest["assignment_manifest_hash"]),
        "--preflight-uid",
        str(candidate_manifest["preflight_uid"]),
        "--command-catalog-hash",
        str(command_catalog_hash_value),
        "--method-id",
        str(assignment["method_id"]),
        "--seed",
        str(assignment["seed"]),
    ]


def validate_candidate_command_catalog(
    config: Mapping[str, object],
    candidate_manifest: Mapping[str, object],
    assignment_manifest: Mapping[str, object],
    catalog: Mapping[str, object],
) -> list[dict[str, str]]:
    blockers: list[dict[str, str]] = []
    if catalog.get("schema_version") != "neuronsifter-operational-twin-candidate-command-catalog-v1":
        blockers.append(_block("candidate_command_schema_mismatch", "Command catalog schema version is invalid."))
    if catalog.get("command_catalog_hash") != command_catalog_hash(catalog):
        blockers.append(_block("candidate_command_catalog_hash_mismatch", "Command catalog hash does not recompute."))
    for field in ("candidate_lineage_uid", "protocol_uid", "preflight_uid"):
        if catalog.get(field) != candidate_manifest.get(field):
            blockers.append(_block("candidate_command_identity_mismatch", f"Command catalog {field} drifted."))
    if catalog.get("candidate_manifest_hash") != candidate_manifest.get("candidate_manifest_hash"):
        blockers.append(_block("candidate_command_manifest_hash_mismatch", "Command catalog candidate manifest hash drifted."))
    if catalog.get("assignment_manifest_hash") != assignment_manifest.get("assignment_manifest_hash"):
        blockers.append(_block("candidate_command_assignment_hash_mismatch", "Command catalog assignment manifest hash drifted."))
    commands = catalog.get("commands")
    if not isinstance(commands, list) or len(commands) != 30:
        blockers.append(_block("candidate_command_count_mismatch", "Command catalog must contain 15 initial and 15 conditional recovery rows."))
        return blockers
    expected_attempts = {
        str(attempt["attempt_uid"]): (assignment, attempt)
        for assignment in assignment_manifest.get("assignments", [])
        for attempt in dict(assignment).get("attempts", [])
    }
    if set(expected_attempts) != {str(dict(row).get("attempt_uid", "")) for row in commands}:
        blockers.append(_block("candidate_command_attempt_coverage_mismatch", "Command rows do not cover exactly the manifest attempt slots."))
    forbidden_flags = {
        "--resume",
        "--resume-input",
        "--history-input",
        "--checkpoint-input",
        "--optimizer-state-input",
        "--model-state-input",
        "--calibration-state-input",
    }
    output_roots: list[str] = []
    initial_count = 0
    recovery_count = 0
    for index, value in enumerate(commands):
        if not isinstance(value, Mapping):
            blockers.append(
                _block(
                    "candidate_command_row_keyset_mismatch",
                    f"Command row {index} is not an object with the exact allowed keyset.",
                )
            )
            continue
        row = dict(value)
        row_keys = set(row)
        if row_keys != CANDIDATE_COMMAND_ROW_KEYS:
            missing = sorted(CANDIDATE_COMMAND_ROW_KEYS - row_keys)
            unexpected = sorted(
                str(key) for key in row_keys - CANDIDATE_COMMAND_ROW_KEYS
            )
            blockers.append(
                _block(
                    "candidate_command_row_keyset_mismatch",
                    f"Command row {index} keyset drifted: missing={missing}; "
                    f"unexpected={unexpected}.",
                )
            )
        argv = row.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(part, str) for part in argv):
            blockers.append(_block("candidate_command_not_structured_argv", f"Command row {index} is not a string argv array."))
            continue
        if row.get("command_index") != index:
            blockers.append(_block("candidate_command_index_mismatch", f"Command row {index} has a noncanonical index."))
        attempt_uid = str(row.get("attempt_uid", ""))
        expected = expected_attempts.get(attempt_uid)
        if expected is None:
            continue
        assignment, attempt = expected
        identity = candidate_identity_payload(dict(assignment), dict(attempt))
        for field, expected_value in identity.items():
            if row.get(field) != expected_value:
                blockers.append(_block("candidate_command_row_identity_mismatch", f"{attempt_uid} {field} drifted."))
        if row.get("output_root") != attempt.get("output_root"):
            blockers.append(_block("candidate_command_output_mismatch", f"{attempt_uid} output root drifted."))
        for field in ("execution_authority", "resource_environment_receipt"):
            if row.get(field) != attempt.get(field):
                blockers.append(
                    _block(
                        "candidate_command_execution_binding_mismatch",
                        f"{attempt_uid} {field} drifted.",
                    )
                )
        output_roots.append(str(row.get("output_root", "")))
        if row.get("attempt_kind") == "initial":
            initial_count += 1
        elif row.get("attempt_kind") == "infrastructure_recovery_01":
            recovery_count += 1
        else:
            blockers.append(_block("candidate_command_attempt_kind_invalid", f"{attempt_uid} attempt kind is invalid."))
        expected_argv = expected_candidate_command_argv(
            config,
            candidate_manifest,
            assignment_manifest,
            assignment,
            attempt,
            command_catalog_hash_value=catalog.get("command_catalog_hash"),
        )
        if argv != expected_argv:
            blockers.append(
                _block(
                    "candidate_command_argv_exact_mismatch",
                    f"{attempt_uid} argv differs from the canonical ordered sequence.",
                )
            )
        if any(flag in argv for flag in forbidden_flags):
            blockers.append(_block("candidate_command_forbidden_state_input", f"{attempt_uid} contains a forbidden state-input flag."))
    if initial_count != 15 or recovery_count != 15:
        blockers.append(_block("candidate_command_attempt_kind_count_mismatch", "Catalog must contain exactly 15 initial and 15 recovery rows."))
    if len(set(output_roots)) != len(output_roots):
        blockers.append(_block("candidate_attempt_output_collision", "Candidate attempt output roots must be disjoint."))
    if catalog.get("training_executed") is not False or catalog.get("calibration_fitting_executed") is not False or catalog.get("checkpoint_created") is not False:
        blockers.append(_block("candidate_command_execution_evidence_present", "Protocol command catalog cannot claim execution evidence."))
    return blockers


def _candidate_open_data_identities(
    protocol: Mapping[str, object], root: Path
) -> tuple[list[dict[str, str]], dict[str, object], dict[str, object]]:
    blockers: list[dict[str, str]] = []
    scientific = dict(protocol["scientific_contract"])
    authority_path = str(scientific["open_data_authority"])
    try:
        authority = read_repo_json(root, authority_path)
    except (OperationalTwinProtocolError, json.JSONDecodeError) as error:
        return [_block("candidate_open_data_authority_unavailable", str(error))], {}, {}
    if authority.get("terminal_decision") != QUALIFIED_TERMINAL or authority.get("training_authorized") is not True:
        blockers.append(_block("candidate_open_data_not_qualified", "Open data is not qualification-authorized."))
    identities: dict[str, object] = {}
    for role in ("training", "development", "calibration"):
        path = str(dict(scientific["manifests"])[role])
        authority_row = dict(dict(authority.get("input_identities", {})).get(role, {}))
        try:
            content = read_repo_bytes(root, path)
            payload = json.loads(content.decode("utf-8"))
            identity = {
                "path": path,
                "sha256": hashlib.sha256(content).hexdigest(),
                "content_identity": payload.get("content_identity"),
            }
            identities[role] = identity
            if authority_row.get("path") != path or authority_row.get("sha256") != identity["sha256"]:
                blockers.append(_block("candidate_open_data_file_identity_mismatch", f"{role} manifest file identity drifted."))
            if dict(authority.get("input_content_identities", {})).get(role) != identity["content_identity"]:
                blockers.append(_block("candidate_open_data_content_identity_mismatch", f"{role} manifest content identity drifted."))
        except (OperationalTwinProtocolError, json.JSONDecodeError) as error:
            blockers.append(_block("candidate_open_data_manifest_unavailable", f"{role}: {error}"))
    return blockers, authority, identities


def build_candidate_preflight_authority(
    config: Mapping[str, object],
    root: Path,
    candidate_manifest: Mapping[str, object],
    assignment_manifest: Mapping[str, object],
    command_catalog: Mapping[str, object],
) -> dict[str, object]:
    protocol = _candidate_protocol(config)
    source_bindings = dict(protocol["source_bindings"])
    source_paths = [
        str(value)
        for key, value in source_bindings.items()
        if key not in {"snapshot_invariant", "snapshot_repair_prospective_only"}
    ]
    open_blockers, open_authority, open_identities = _candidate_open_data_identities(protocol, root)
    if open_blockers:
        raise OperationalTwinProtocolError(
            "; ".join(f"{row['code']}: {row['message']}" for row in open_blockers)
        )
    historical_path = str(dict(protocol["historical_boundary"])["audit_path"])
    schema_path = "Libs/configs/schema/operational_twin_candidate_lineage_protocol.schema.yaml"
    config_path = "Libs/configs/model/operational_twin_candidate_lineage_protocol.yaml"
    config_identity = repo_file_identity(root, config_path)
    schema_identity = repo_file_identity(root, schema_path)
    source_identities = [repo_file_identity(root, path) for path in source_paths]
    execution_authority_contract = dict(protocol["execution_authority_contract"])
    payload: dict[str, object] = {
        "schema_version": CANDIDATE_PREFLIGHT_SCHEMA_VERSION,
        "status": "PASS",
        "terminal_decision": CANDIDATE_POSITIVE_TERMINAL,
        "authoritative_date": protocol["authoritative_date"],
        "preflight_uid": compute_candidate_preflight_uid(protocol),
        "candidate_lineage_uid": protocol["candidate_lineage_uid"],
        "protocol_uid": protocol["protocol_uid"],
        "candidate_manifest_hash": candidate_manifest["candidate_manifest_hash"],
        "assignment_manifest_hash": assignment_manifest["assignment_manifest_hash"],
        "command_catalog_hash": command_catalog["command_catalog_hash"],
        "config_identity": config_identity,
        "config_hash": canonical_hash(protocol),
        "schema_identity": schema_identity,
        "scientific_contract_hash": canonical_hash(protocol["scientific_contract"]),
        "historical_v1_scientific_contract_hash": canonical_hash(
            _candidate_scientific_contract_from_v1(config)
        ),
        "open_data_authority_identity": repo_file_identity(
            root, str(dict(protocol["scientific_contract"])["open_data_authority"])
        ),
        "open_data_terminal": open_authority.get("terminal_decision"),
        "open_data_identities": open_identities,
        "sealed_manifest_metadata": protocol["sealed_manifest_metadata"],
        "scheduler_evidence_contract": protocol["scheduler_evidence_contract"],
        "supersession_contract": protocol["supersession_contract"],
        "supersession_contract_hash": candidate_supersession_contract_hash(protocol),
        "historical_authority_identity": repo_file_identity(root, historical_path),
        "source_identities": source_identities,
        "source_identity_hash": canonical_hash(
            {
                "config_identity": config_identity,
                "schema_identity": schema_identity,
                "source_identities": source_identities,
            }
        ),
        "round_41_snapshot_contract": {
            "invariant": source_bindings["snapshot_invariant"],
            "prospective_only": True,
            "production_regression": "test_selected_checkpoint_snapshot_survives_later_training_and_calibration",
            "required": True,
        },
        "method_ids": REQUIRED_METHOD_IDS,
        "seeds": [360101, 360211, 360307],
        "assignment_count": 15,
        "attempt_slot_count": 30,
        "resource_execution_gate": {
            "fresh_sinfo_required_before_dispatch": True,
            "fresh_resource_evidence_required_before_dispatch": True,
            "new_resource_environment_receipt_required_before_dispatch": True,
            "independent_execution_authority_required_before_dispatch": True,
            "approved_protocol_review_path": execution_authority_contract[
                "approved_protocol_review_path"
            ],
            "candidate_execution_authority_path_template": execution_authority_contract[
                "candidate_execution_authority_path_template"
            ],
            "resource_environment_receipt_path_template": execution_authority_contract[
                "resource_environment_receipt_path_template"
            ],
            "authority_schema_version": execution_authority_contract[
                "authority_schema_version"
            ],
            "resource_receipt_schema_version": execution_authority_contract[
                "resource_receipt_schema_version"
            ],
            "scheduler_evidence_contract_version": execution_authority_contract[
                "scheduler_evidence_contract_version"
            ],
            "fresh_sinfo_command": execution_authority_contract[
                "fresh_sinfo_command"
            ],
            "fresh_resource_command": execution_authority_contract[
                "fresh_resource_command"
            ],
            "authority_bound_in_this_round": False,
            "resource_receipt_bound_in_this_round": False,
            "trainable_placement": {
                "profile": "gpu_single",
                **CANDIDATE_RESOURCE_PROFILES["gpu_single"],
            },
            "no_fit_placement": {
                "profile": "cpu_light",
                **CANDIDATE_RESOURCE_PROFILES["cpu_light"],
            },
        },
        "attempt_roots_absent_at_freeze": True,
        "access_counters": protocol["access_counters"],
        "closed_flags": protocol["closed_flags"],
        "training_executed": False,
        "calibration_fitting_executed": False,
        "checkpoint_created": False,
        "sufficient_training": False,
        "twin_qualification_pass": False,
        "formal_qualification": False,
        "presealed_freeze_created": False,
    }
    return payload


def validate_candidate_artifacts(
    config: Mapping[str, object],
    root: Path,
    candidate_manifest: Mapping[str, object],
    assignment_manifest: Mapping[str, object],
    command_catalog: Mapping[str, object],
    preflight: Mapping[str, object],
    *,
    require_attempt_roots_absent: bool,
    resolved_context: ResolvedCandidateExecutionProtocol | None = None,
) -> list[dict[str, str]]:
    if resolved_context is not None:
        expected = load_candidate_protocol_artifacts(config, root, resolved_context=resolved_context)
        observed = (candidate_manifest, assignment_manifest, command_catalog, preflight)
        if any(dict(actual) != reference for actual, reference in zip(observed, expected, strict=True)):
            return [_block("candidate_authenticated_artifact_drift", "Artifact differs from the authenticated frozen package.")]
        blockers = []
        for assignment in assignment_manifest["assignments"]:
            for attempt in assignment["attempts"]:
                try:
                    candidate_attempt_path(root, str(attempt["output_root"]), require_absent=require_attempt_roots_absent)
                except OperationalTwinProtocolError as error:
                    blockers.append(_block("candidate_attempt_root_invalid", str(error)))
        return blockers
    blockers = validate_candidate_protocol_config(config, root)
    expected_candidate = build_candidate_lineage_manifest(config)
    if candidate_manifest != expected_candidate:
        blockers.append(_block("candidate_manifest_drift", "Candidate lineage manifest does not recompute exactly."))
    expected_assignment = build_candidate_assignment_manifest(config, expected_candidate)
    if assignment_manifest != expected_assignment:
        blockers.append(_block("candidate_assignment_manifest_drift", "Candidate assignment manifest does not recompute exactly."))
    blockers.extend(
        validate_candidate_command_catalog(
            config, candidate_manifest, assignment_manifest, command_catalog
        )
    )
    try:
        expected_preflight = build_candidate_preflight_authority(
            config,
            root,
            expected_candidate,
            expected_assignment,
            command_catalog,
        )
        if preflight != expected_preflight:
            blockers.append(_block("candidate_preflight_drift", "Candidate preflight does not recompute exactly."))
    except OperationalTwinProtocolError as error:
        blockers.append(_block("candidate_preflight_authority_unavailable", str(error)))
    if preflight.get("status") != "PASS" or preflight.get("terminal_decision") != CANDIDATE_POSITIVE_TERMINAL:
        blockers.append(_block("candidate_preflight_not_frozen", "Candidate preflight is not the exact positive terminal."))
    for assignment in assignment_manifest.get("assignments", []):
        for attempt in dict(assignment).get("attempts", []):
            declared_root = str(dict(attempt).get("output_root", ""))
            try:
                candidate_attempt_path(
                    root,
                    declared_root,
                    require_absent=require_attempt_roots_absent,
                )
            except OperationalTwinProtocolError as error:
                blockers.append(
                    _block(
                        "candidate_attempt_root_invalid",
                        f"{declared_root}: {error}",
                    )
                )
    return blockers


def load_candidate_protocol_artifacts(
    config: Mapping[str, object], root: Path,
    resolved_context: ResolvedCandidateExecutionProtocol | None = None,
) -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    if resolved_context is not None:
        return tuple(deepcopy(dict(value)) for value in (
            resolved_context.candidate_manifest, resolved_context.assignment_manifest,
            resolved_context.command_catalog, resolved_context.preflight,
        ))
    paths = dict(_candidate_protocol(config)["output_paths"])
    return (
        read_repo_json(root, paths["candidate_lineage_manifest"]),
        read_repo_json(root, paths["assignment_manifest"]),
        read_repo_json(root, paths["assignment_commands"]),
        read_repo_json(root, paths["preflight_authority"]),
    )


def _exact_candidate_execution_bindings(
    execution_authority_binding: object,
    resource_receipt_binding: object,
    resolved_context: ResolvedCandidateExecutionProtocol | None = None,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    """Return candidate bindings only when both use their exact frozen schemas."""
    authority = None
    if (
        isinstance(execution_authority_binding, Mapping)
        and set(execution_authority_binding)
        == candidate_execution_binding_keys(CANDIDATE_EXECUTION_AUTHORITY_BINDING_KEYS, resolved_context)
    ):
        authority = dict(execution_authority_binding)
    receipt = None
    if (
        isinstance(resource_receipt_binding, Mapping)
        and set(resource_receipt_binding)
        == candidate_execution_binding_keys(CANDIDATE_RESOURCE_RECEIPT_BINDING_KEYS, resolved_context)
    ):
        receipt = dict(resource_receipt_binding)
    if authority is not None:
        require_successor_execution_binding(authority, resolved_context)
    if receipt is not None:
        require_successor_execution_binding(receipt, resolved_context)
    if resolved_context is not None and authority is not None and receipt is not None:
        if (not isinstance(authority.get("runtime_dispatch_binding"), Mapping)
                or authority["runtime_dispatch_binding"] != receipt.get("runtime_dispatch_binding")):
            raise OperationalTwinProtocolError("candidate_execution_runtime_dispatch_binding_mismatch")
    return authority, receipt


def require_candidate_execution_authority(
    config: Mapping[str, object],
    root: Path,
    *,
    assignment: Mapping[str, object],
    attempt: Mapping[str, object],
    candidate_manifest: Mapping[str, object],
    assignment_manifest: Mapping[str, object],
    command_catalog: Mapping[str, object],
    preflight: Mapping[str, object],
    execution_authority_path: str | Path,
    resource_environment_receipt_path: str | Path,
    cuda_available: bool,
    cuda_device_count: int,
    torch_version: str,
    current_source_commit: str | None = None,
    now: datetime | None = None,
    validate_current_runtime: bool = True,
    resolved_context: ResolvedCandidateExecutionProtocol | None = None,
) -> dict[str, object]:
    """Require later-round approval and a fresh assignment-bound resource receipt."""
    protocol = _candidate_protocol(config, resolved_context)
    contract = _candidate_execution_contract(protocol, resolved_context)
    expected_paths = candidate_execution_artifact_paths(protocol, assignment, attempt)
    if resolved_context is not None:
        from Libs.utils.evaluation.operational_twin_successor_execution_handoff import _path
        for value in (execution_authority_path, resource_environment_receipt_path):
            candidate_path = Path(value)
            relative = candidate_path.relative_to(root).as_posix() if candidate_path.is_absolute() else os.fspath(value)
            _path(root, relative)
        for value in expected_paths.values():
            _path(root, str(value))
    supplied_authority_path = repo_path(root, execution_authority_path, must_exist=False)
    supplied_receipt_path = repo_path(
        root,
        resource_environment_receipt_path,
        must_exist=False,
    )
    expected_authority_path = repo_path(
        root,
        expected_paths["execution_authority"],
        must_exist=False,
    )
    expected_receipt_path = repo_path(
        root,
        expected_paths["resource_environment_receipt"],
        must_exist=False,
    )
    if supplied_authority_path != expected_authority_path:
        raise OperationalTwinProtocolError(
            "candidate_execution_authority_path_mismatch: authority path differs from the frozen attempt contract."
        )
    if supplied_receipt_path != expected_receipt_path:
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_path_mismatch: receipt path differs from the frozen attempt contract."
        )
    if not supplied_authority_path.is_file():
        raise OperationalTwinProtocolError(
            "candidate_execution_authority_missing: independently approved execution authority is absent."
        )
    if not supplied_receipt_path.is_file():
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_missing: fresh resource/environment receipt is absent."
        )

    try:
        authority = json.loads(supplied_authority_path.read_text(encoding="utf-8"))
        receipt = json.loads(supplied_receipt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise OperationalTwinProtocolError(
            "candidate_execution_authority_invalid: authority or resource receipt is not valid JSON."
        ) from error
    if not isinstance(authority, dict) or set(authority) != candidate_execution_binding_keys(CANDIDATE_EXECUTION_AUTHORITY_KEYS, resolved_context):
        raise OperationalTwinProtocolError(
            "candidate_execution_authority_schema_invalid: authority keyset is not exact."
        )
    if not isinstance(receipt, dict) or set(receipt) != candidate_execution_binding_keys(CANDIDATE_RESOURCE_RECEIPT_KEYS, resolved_context):
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_schema_invalid: receipt keyset is not exact."
        )
    require_successor_execution_binding(authority, resolved_context)
    require_successor_execution_binding(receipt, resolved_context)
    slurm = receipt.get("slurm")
    environment = receipt.get("environment")
    if not isinstance(slurm, Mapping) or set(slurm) != CANDIDATE_RESOURCE_SLURM_KEYS:
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_schema_invalid: slurm keyset is not exact."
        )
    if (
        not isinstance(environment, Mapping)
        or set(environment) != CANDIDATE_RESOURCE_ENVIRONMENT_KEYS
    ):
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_schema_invalid: environment keyset is not exact."
        )
    allocation = receipt.get("resource_allocation")
    if (
        not isinstance(allocation, Mapping)
        or set(allocation) != CANDIDATE_RESOURCE_ALLOCATION_KEYS
    ):
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_schema_invalid: resource allocation keyset is not exact."
        )

    resource_profile = str(assignment.get("resource_profile", ""))
    expected_requirements = CANDIDATE_RESOURCE_PROFILES.get(resource_profile)
    if expected_requirements is None or assignment.get(
        "resource_requirements"
    ) != expected_requirements:
        raise OperationalTwinProtocolError(
            "candidate_resource_requirements_mismatch: assignment resource requirements differ from the frozen profile."
        )

    expected_identity = {
        **candidate_identity_payload(assignment, attempt),
        "method_id": assignment.get("method_id"),
        "seed": assignment.get("seed"),
        "output_root": attempt.get("output_root"),
        "resource_profile": assignment.get("resource_profile"),
        "resource_requirements": assignment.get("resource_requirements"),
        "candidate_manifest_hash": candidate_manifest.get("candidate_manifest_hash"),
        "assignment_manifest_hash": assignment_manifest.get("assignment_manifest_hash"),
        "command_catalog_hash": command_catalog.get("command_catalog_hash"),
        "preflight_uid": preflight.get("preflight_uid"),
    }
    for field, expected_value in expected_identity.items():
        if receipt.get(field) != expected_value:
            raise OperationalTwinProtocolError(
                f"candidate_resource_receipt_identity_mismatch: {field} drifted."
            )
        if authority.get(field) != expected_value:
            raise OperationalTwinProtocolError(
                f"candidate_execution_authority_identity_mismatch: {field} drifted."
            )

    source_commit = current_source_commit or _current_git_commit(root)
    _require_executing_commit(root, source_commit, resolved_context)
    dispatch_binding = None
    if resolved_context is not None:
        # Dispatch publication lives above these reusable predecessor predicates.
        from Libs.utils.evaluation.operational_twin_candidate_execution import require_candidate_dispatch_binding
        dispatch_binding = require_candidate_dispatch_binding(
            config, root, resolved_context, assignment=assignment, attempt=attempt,
            slurm=slurm, source_commit=source_commit,
            validate_current_runtime=validate_current_runtime,
        )
        if any(payload.get("runtime_dispatch_binding") != dispatch_binding for payload in (authority, receipt)):
            raise OperationalTwinProtocolError("candidate_execution_runtime_dispatch_binding_mismatch")
    source_identity_hash = candidate_source_identity_hash(preflight)
    if receipt.get("source_commit") != source_commit:
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_source_commit_mismatch: receipt does not bind the executing source commit."
        )
    if authority.get("source_commit") != source_commit:
        raise OperationalTwinProtocolError(
            "candidate_execution_authority_source_commit_mismatch: authority does not bind the executing source commit."
        )
    if receipt.get("source_identity_hash") != source_identity_hash:
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_source_identity_mismatch: receipt does not bind the frozen source identities."
        )
    if authority.get("source_identity_hash") != source_identity_hash:
        raise OperationalTwinProtocolError(
            "candidate_execution_authority_source_identity_mismatch: authority does not bind the frozen source identities."
        )

    if receipt.get("schema_version") != contract["resource_receipt_schema_version"]:
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_schema_invalid: schema version drifted."
        )
    if receipt.get("status") != "PASS" or receipt.get("terminal_decision") != contract[
        "positive_resource_receipt_terminal"
    ]:
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_not_valid: receipt lacks the exact positive terminal."
        )
    if receipt.get("receipt_uid") != compute_candidate_resource_receipt_uid(receipt):
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_uid_mismatch: receipt UID does not recompute."
        )
    check_time = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    created_at = _parse_utc_timestamp(receipt.get("created_at"), label="created_at")
    sinfo_captured_at = _parse_utc_timestamp(
        receipt.get("sinfo_captured_at"),
        label="sinfo_captured_at",
    )
    resource_captured_at = _parse_utc_timestamp(
        receipt.get("resource_captured_at"),
        label="resource_captured_at",
    )
    maximum_age = int(contract["maximum_resource_receipt_age_seconds"])
    if any(
        timestamp > check_time
        for timestamp in (created_at, sinfo_captured_at, resource_captured_at)
    ):
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_from_future: receipt timestamps are later than runtime."
        )
    if created_at < max(sinfo_captured_at, resource_captured_at):
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_timestamp_order_invalid: receipt predates its scheduler evidence."
        )
    if (
        (check_time - created_at).total_seconds() > maximum_age
        or (check_time - sinfo_captured_at).total_seconds() > maximum_age
        or (check_time - resource_captured_at).total_seconds() > maximum_age
    ):
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_stale: receipt and sinfo capture must be fresh at dispatch."
        )
    expected_sinfo_command = [
        str(value).format(
            nodelist=dict(assignment["resource_requirements"])["nodelist"]
        )
        for value in contract["fresh_sinfo_command"]
    ]
    if receipt.get("fresh_sinfo") is not True or receipt.get("sinfo_command") != expected_sinfo_command:
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_sinfo_invalid: fresh sinfo evidence or command drifted."
        )
    sinfo_stdout = receipt.get("sinfo_stdout")
    if not isinstance(sinfo_stdout, str) or not sinfo_stdout.strip():
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_sinfo_invalid: sinfo output is empty."
        )
    if receipt.get("sinfo_stdout_sha256") != sha256_text(sinfo_stdout):
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_sinfo_hash_mismatch: sinfo output hash drifted."
        )

    resource_requirements = dict(expected_requirements)
    expected_node = str(resource_requirements["nodelist"])
    expected_partition = str(resource_requirements["partition"])
    _require_candidate_resource_evidence(
        receipt,
        resource_requirements,
        contract["fresh_resource_command"],
    )
    if validate_current_runtime:
        runtime_hostname = socket.gethostname().split(".", 1)[0]
        runtime_slurm = candidate_runtime_slurm_identity(hostname=runtime_hostname)
        if dict(slurm) != runtime_slurm:
            raise OperationalTwinProtocolError(
                "candidate_resource_receipt_runtime_mismatch: receipt scheduler fields differ from the live Slurm environment."
            )
        expected_environment = _runtime_environment_identity(
            hostname=runtime_hostname,
            cuda_available=cuda_available,
            cuda_device_count=cuda_device_count,
            torch_version=torch_version,
        )
        if dict(environment) != expected_environment:
            raise OperationalTwinProtocolError(
                "candidate_resource_receipt_environment_mismatch: receipt environment differs from runtime."
            )
    _require_live_candidate_resource_placement(
        resource_requirements,
        slurm,
        environment,
    )
    sinfo_rows = [line.split("|") for line in sinfo_stdout.splitlines()]
    matching_sinfo_rows = [
        row
        for row in sinfo_rows
        if len(row) == 4
        and row[0] == expected_node
        and row[1].rstrip("*") == expected_partition
    ]
    if len(matching_sinfo_rows) != 1:
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_sinfo_invalid: sinfo evidence does not identify the exact node and partition once."
        )
    node_state = matching_sinfo_rows[0][2].lower().rstrip("*~#+")
    if any(token in node_state for token in ("down", "drain", "fail", "maint")):
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_sinfo_invalid: required node is not execution-capable."
        )
    if resource_requirements["gres"] == "gpu:1" and "gpu" not in matching_sinfo_rows[0][3].lower():
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_sinfo_invalid: gpu03 evidence does not advertise a GPU."
        )
    if any(receipt.get(field) is not False for field in (
        "training_executed",
        "calibration_fitting_executed",
        "checkpoint_created",
    )):
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_execution_evidence_present: receipt must predate execution."
        )
    if receipt.get("access_counters") != protocol["access_counters"]:
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_access_mismatch: access counters drifted."
        )

    review_path = repo_path(
        root,
        str(contract["approved_protocol_review_path"]),
        must_exist=True,
    )
    review = json.loads(review_path.read_text(encoding="utf-8"))
    required_review = {
        "schema_version": contract["required_review_schema_version"],
        "round_id": contract["required_review_round_id"],
        "direction_id": contract["required_review_direction_id"],
        "decision": contract["required_review_decision"],
    }
    for field, expected_value in required_review.items():
        if review.get(field) != expected_value:
            raise OperationalTwinProtocolError(
                f"candidate_execution_review_not_approved: review {field} drifted."
            )
    review_identity = repo_file_identity(root, review_path)
    if (root / ".git").exists() and review_identity.get("git_object_id") is None:
        raise OperationalTwinProtocolError(
            "candidate_execution_review_not_committed: approved Round 45 review must exist in the executing Git tree."
        )
    receipt_identity = candidate_runtime_artifact_identity(root, supplied_receipt_path, resolved_context)
    if authority.get("schema_version") != contract["authority_schema_version"]:
        raise OperationalTwinProtocolError(
            "candidate_execution_authority_schema_invalid: schema version drifted."
        )
    if authority.get("status") != "PASS" or authority.get("terminal_decision") != contract[
        "positive_authority_terminal"
    ]:
        raise OperationalTwinProtocolError(
            "candidate_execution_authority_not_approved: authority lacks the exact positive terminal."
        )
    if authority.get("authority_role") != contract["authority_role"]:
        raise OperationalTwinProtocolError(
            "candidate_execution_authority_role_mismatch: authority role drifted."
        )
    if authority.get("authority_uid") != compute_candidate_execution_authority_uid(authority):
        raise OperationalTwinProtocolError(
            "candidate_execution_authority_uid_mismatch: authority UID does not recompute."
        )
    issued_at = _parse_utc_timestamp(authority.get("issued_at"), label="issued_at")
    if issued_at > check_time or issued_at < created_at:
        raise OperationalTwinProtocolError(
            "candidate_execution_authority_timestamp_invalid: authority must be issued after its resource receipt and before execution."
        )
    if authority.get("approved_protocol_review_decision") != "approved":
        raise OperationalTwinProtocolError(
            "candidate_execution_review_not_approved: authority does not bind an approved review."
        )
    if authority.get("approved_protocol_review_identity") != review_identity:
        raise OperationalTwinProtocolError(
            "candidate_execution_review_identity_mismatch: approved review identity drifted."
        )
    if authority.get("resource_environment_receipt_identity") != {
        **receipt_identity,
        "receipt_uid": receipt["receipt_uid"],
    }:
        raise OperationalTwinProtocolError(
            "candidate_execution_resource_receipt_identity_mismatch: authority does not bind the exact receipt."
        )
    if authority.get("fresh_initialization_receipt") != candidate_fresh_initialization_receipt(protocol, resolved_context):
        raise OperationalTwinProtocolError(
            "candidate_execution_fresh_initialization_mismatch: authority does not require the frozen fresh initialization."
        )
    expected_authorizations = {
        "state_resume_allowed": False,
        "training_execution_authorized": True,
        "checkpoint_creation_authorized": assignment.get(
            "checkpoint_creation_allowed"
        ),
        "calibration_fitting_authorized": assignment.get("calibration_eligible"),
        "access_counters": protocol["access_counters"],
    }
    for field, expected_value in expected_authorizations.items():
        if authority.get(field) != expected_value:
            raise OperationalTwinProtocolError(
                f"candidate_execution_authority_invalid: {field} drifted."
            )

    execution_authority_binding = {
            **candidate_runtime_artifact_identity(root, supplied_authority_path, resolved_context),
            "authority_uid": authority["authority_uid"],
            "source_commit": source_commit,
            "approved_protocol_review_identity": review_identity,
        }
    resource_receipt_binding = {
            **receipt_identity,
            "receipt_uid": receipt["receipt_uid"],
            "source_commit": source_commit,
            "scheduler": dict(slurm),
            "environment": dict(environment),
            "resource_allocation": dict(allocation),
        }
    if resolved_context is not None:
        execution_authority_binding["successor_execution_binding"] = dict(resolved_context.binding)
        resource_receipt_binding["successor_execution_binding"] = dict(resolved_context.binding)
        execution_authority_binding["runtime_dispatch_binding"] = deepcopy(dispatch_binding)
        resource_receipt_binding["runtime_dispatch_binding"] = deepcopy(dispatch_binding)
    checked_authority_binding, checked_receipt_binding = (
        _exact_candidate_execution_bindings(
            execution_authority_binding,
            resource_receipt_binding,
            resolved_context=resolved_context,
        )
    )
    if checked_authority_binding is None:
        raise OperationalTwinProtocolError(
            "candidate_execution_authority_binding_invalid: runtime authority binding keyset drifted."
        )
    if checked_receipt_binding is None:
        raise OperationalTwinProtocolError(
            "candidate_resource_receipt_binding_invalid: runtime receipt binding keyset drifted."
        )
    return {
        "execution_authority": checked_authority_binding,
        "resource_environment_receipt": checked_receipt_binding,
        "resource_receipt_payload": receipt,
        "resolved_protocol": resolved_context,
    }


def _validate_retained_candidate_execution_bindings(
    protocol: Mapping[str, object],
    root: Path,
    assignment: Mapping[str, object],
    predecessor_attempt: Mapping[str, object],
    status: Mapping[str, object],
    candidate_manifest: Mapping[str, object],
    assignment_manifest: Mapping[str, object],
    command_catalog: Mapping[str, object],
    preflight: Mapping[str, object],
    resolved_context: ResolvedCandidateExecutionProtocol | None = None,
) -> list[dict[str, str]]:
    """Validate frozen predecessor authority at its original execution boundary."""
    blockers: list[dict[str, str]] = []
    authority_value = status.get("candidate_execution_authority")
    receipt_value = status.get("candidate_resource_environment_receipt")
    authority_binding, receipt_binding = _exact_candidate_execution_bindings(
        authority_value,
        receipt_value,
        resolved_context=resolved_context,
    )
    if authority_binding is None:
        code = (
            "candidate_recovery_predecessor_authority_binding_invalid"
            if isinstance(authority_value, Mapping)
            else "candidate_recovery_predecessor_authority_missing"
        )
        blockers.append(
            _block(
                code,
                "Recovery predecessor execution-authority binding is absent or its keyset is not exact.",
            )
        )
    if receipt_binding is None:
        code = (
            "candidate_recovery_predecessor_resource_receipt_binding_invalid"
            if isinstance(receipt_value, Mapping)
            else "candidate_recovery_predecessor_resource_receipt_missing"
        )
        blockers.append(
            _block(
                code,
                "Recovery predecessor resource/environment receipt binding is absent or its keyset is not exact.",
            )
        )
    if authority_binding is None or receipt_binding is None:
        return blockers

    try:
        expected_paths = candidate_execution_artifact_paths(
            protocol,
            assignment,
            predecessor_attempt,
        )
    except (KeyError, OperationalTwinProtocolError) as error:
        blockers.append(
            _block(
                "candidate_recovery_predecessor_execution_contract_invalid",
                f"Recovery predecessor execution contract is unavailable: {error}",
            )
        )
        return blockers
    for field, binding in (
        ("execution_authority", authority_binding),
        ("resource_environment_receipt", receipt_binding),
    ):
        expected_path = expected_paths[field]
        if (
            predecessor_attempt.get(field) != expected_path
            or binding.get("path") != expected_path
        ):
            blockers.append(
                _block(
                    "candidate_recovery_predecessor_artifact_path_mismatch",
                    f"Recovery predecessor {field} path is not manifest-derived.",
                )
            )

    frozen_assignments = assignment_manifest.get("assignments")
    matching_assignments = (
        [
            dict(row)
            for row in frozen_assignments
            if isinstance(row, Mapping)
            and row.get("assignment_uid") == assignment.get("assignment_uid")
        ]
        if isinstance(frozen_assignments, list)
        else []
    )
    if len(matching_assignments) != 1 or matching_assignments[0] != dict(assignment):
        blockers.append(
            _block(
                "candidate_recovery_predecessor_assignment_manifest_mismatch",
                "Recovery predecessor assignment is not the exact frozen manifest row.",
            )
        )
    else:
        frozen_attempts = matching_assignments[0].get("attempts")
        matching_attempts = (
            [
                dict(row)
                for row in frozen_attempts
                if isinstance(row, Mapping)
                and row.get("attempt_uid")
                == predecessor_attempt.get("attempt_uid")
            ]
            if isinstance(frozen_attempts, list)
            else []
        )
        if (
            len(matching_attempts) != 1
            or matching_attempts[0] != dict(predecessor_attempt)
        ):
            blockers.append(
                _block(
                    "candidate_recovery_predecessor_attempt_manifest_mismatch",
                    "Recovery predecessor attempt is not the exact frozen manifest row.",
                )
            )

    expected_initialization = candidate_fresh_initialization_receipt(protocol, resolved_context)
    if (
        predecessor_attempt.get("fresh_random_initialization") is not True
        or predecessor_attempt.get("state_resume_allowed") is not False
        or status.get("fresh_initialization_receipt") != expected_initialization
        or status.get("state_resume_used") is not False
    ):
        blockers.append(
            _block(
                "candidate_recovery_predecessor_initialization_mismatch",
                "Recovery predecessor does not prove fresh initialization with no state resume.",
            )
        )

    source_commit = receipt_binding.get("source_commit")
    if not isinstance(source_commit, str) or not source_commit.strip():
        blockers.append(
            _block(
                "candidate_recovery_predecessor_source_commit_mismatch",
                "Recovery predecessor source commit is absent.",
            )
        )
        return blockers
    try:
        predecessor_started_at = _parse_utc_timestamp(
            status.get("started_at"),
            label="predecessor_started_at",
        )
        # Successor dispatch validation needs the canonical full stack; the
        # historical predecessor path keeps its original protocol-only contract.
        if resolved_context is not None:
            from Libs.utils.exp_utils import load_config_stack
            retained_config = load_config_stack(root / "Libs/configs/config_stack.yaml")
        else:
            retained_config = {CANDIDATE_PROTOCOL_KEY: protocol}
        validated = require_candidate_execution_authority(
            retained_config,
            root,
            assignment=assignment,
            attempt=predecessor_attempt,
            candidate_manifest=candidate_manifest,
            assignment_manifest=assignment_manifest,
            command_catalog=command_catalog,
            preflight=preflight,
            execution_authority_path=expected_paths["execution_authority"],
            resource_environment_receipt_path=expected_paths[
                "resource_environment_receipt"
            ],
            cuda_available=False,
            cuda_device_count=0,
            torch_version="",
            current_source_commit=source_commit,
            now=predecessor_started_at,
            validate_current_runtime=False,
            resolved_context=resolved_context,
        )
    except (KeyError, TypeError, ValueError, OperationalTwinProtocolError) as error:
        blockers.append(
            _block(
                "candidate_recovery_predecessor_binding_invalid",
                str(error),
            )
        )
        return blockers

    if validated["execution_authority"] != authority_binding:
        blockers.append(
            _block(
                "candidate_recovery_predecessor_authority_binding_mismatch",
                "Recovery predecessor authority binding does not match its frozen file identity.",
            )
        )
    if validated["resource_environment_receipt"] != receipt_binding:
        blockers.append(
            _block(
                "candidate_recovery_predecessor_resource_receipt_binding_mismatch",
                "Recovery predecessor receipt binding does not match its frozen file identity.",
            )
        )
    receipt_payload = dict(validated["resource_receipt_payload"])
    if status.get("scheduler_identity") != receipt_payload.get("slurm"):
        blockers.append(
            _block(
                "candidate_recovery_predecessor_scheduler_mismatch",
                "Recovery predecessor terminal scheduler identity differs from the bound receipt.",
            )
        )
    return blockers


def validate_candidate_recovery_predecessor(
    protocol: Mapping[str, object],
    assignment: Mapping[str, object],
    predecessor_attempt: Mapping[str, object],
    status: Mapping[str, object],
    *,
    allowed_failure_class: object,
    root: Path | None = None,
    candidate_manifest: Mapping[str, object] | None = None,
    assignment_manifest: Mapping[str, object] | None = None,
    command_catalog: Mapping[str, object] | None = None,
    preflight: Mapping[str, object] | None = None,
    resolved_context: ResolvedCandidateExecutionProtocol | None = None,
) -> list[dict[str, str]]:
    """Authenticate the complete failed-attempt identity before one recovery."""
    blockers: list[dict[str, str]] = []
    if not isinstance(status, Mapping):
        return [
            _block(
                "candidate_recovery_predecessor_schema_invalid",
                "Recovery predecessor terminal must be a mapping.",
            ),
            _block(
                "candidate_recovery_not_eligible",
                "Recovery requires the exact same failed assignment and fresh-initialization lineage.",
            ),
        ]
    if set(status) != CANDIDATE_FAILED_TERMINAL_KEYS:
        blockers.append(
            _block(
                "candidate_recovery_predecessor_schema_invalid",
                "Recovery predecessor terminal keyset is not exact.",
            )
        )
    expected_values = {
        "schema_version": "neuronsifter-operational-twin-assignment-terminal-status-v1",
        **candidate_identity_payload(assignment, predecessor_attempt),
        "method_id": assignment.get("method_id"),
        "seed": assignment.get("seed"),
        "output_root": predecessor_attempt.get("output_root"),
        "status": "FAILED",
        "failure_class": allowed_failure_class,
        "fresh_initialization_receipt": candidate_fresh_initialization_receipt(
            protocol, resolved_context
        ),
        "state_resume_used": False,
    }
    for field, expected_value in expected_values.items():
        if status.get(field) != expected_value:
            blockers.append(
                _block(
                    "candidate_recovery_predecessor_identity_mismatch",
                    f"Recovery predecessor {field} drifted.",
                )
            )
    if root is None or any(
        not isinstance(artifact, Mapping)
        for artifact in (
            candidate_manifest,
            assignment_manifest,
            command_catalog,
            preflight,
        )
    ):
        blockers.append(
            _block(
                "candidate_recovery_predecessor_context_missing",
                "Recovery predecessor validation requires the frozen candidate artifacts and repository root.",
            )
        )
    else:
        blockers.extend(
            _validate_retained_candidate_execution_bindings(
                protocol,
                root,
                assignment,
                predecessor_attempt,
                status,
                candidate_manifest,
                assignment_manifest,
                command_catalog,
                preflight,
                resolved_context=resolved_context,
            )
        )
    if blockers:
        blockers.append(
            _block(
                "candidate_recovery_not_eligible",
                "Recovery requires the exact same failed assignment and fresh-initialization lineage.",
            )
        )
    return blockers


def _candidate_directory_open_flags() -> int:
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise OperationalTwinProtocolError(
            "candidate_attempt_nofollow_unavailable: this platform cannot safely claim candidate attempt roots."
        )
    return (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )


def _open_or_create_candidate_ancestor(parent_fd: int, component: str) -> int:
    flags = _candidate_directory_open_flags()
    try:
        return os.open(component, flags, dir_fd=parent_fd)
    except FileNotFoundError:
        try:
            os.mkdir(component, mode=0o755, dir_fd=parent_fd)
        except FileExistsError:
            pass
        try:
            return os.open(component, flags, dir_fd=parent_fd)
        except OSError as error:
            raise OperationalTwinProtocolError(
                "candidate_attempt_path_unsafe: a candidate attempt ancestor changed during claim."
            ) from error
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise OperationalTwinProtocolError(
                "candidate_attempt_path_unsafe: candidate attempt ancestors must be real directories."
            ) from error
        raise


def _remove_created_candidate_leaf(
    parent_fd: int,
    leaf: str,
    created_identity: os.stat_result | None,
) -> None:
    if created_identity is None:
        return
    try:
        current = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
    except OSError:
        return
    if (
        stat.S_ISDIR(current.st_mode)
        and current.st_dev == created_identity.st_dev
        and current.st_ino == created_identity.st_ino
    ):
        try:
            os.rmdir(leaf, dir_fd=parent_fd)
        except OSError:
            pass


def _claim_candidate_attempt_root(root: Path, declared_output_root: str | Path) -> Path:
    """Atomically claim one candidate leaf through no-follow directory handles."""
    output_root = candidate_attempt_path(
        root,
        declared_output_root,
        require_absent=False,
    )
    declared_path = Path(os.fspath(declared_output_root))
    flags = _candidate_directory_open_flags()
    opened_fds: list[int] = []
    leaf_created = False
    created_identity: os.stat_result | None = None
    parent_fd = os.open(root.resolve(strict=True), flags)
    opened_fds.append(parent_fd)
    try:
        for component in declared_path.parts[:-1]:
            parent_fd = _open_or_create_candidate_ancestor(parent_fd, component)
            opened_fds.append(parent_fd)

        leaf = declared_path.parts[-1]
        try:
            os.mkdir(leaf, mode=0o755, dir_fd=parent_fd)
            leaf_created = True
        except FileExistsError as error:
            raise OperationalTwinProtocolError(
                "candidate_attempt_output_collision: the manifest-bound attempt root already exists."
            ) from error
        except OSError as error:
            if error.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise OperationalTwinProtocolError(
                    "candidate_attempt_path_unsafe: the candidate attempt leaf cannot be created safely."
                ) from error
            raise

        leaf_fd = os.open(leaf, flags, dir_fd=parent_fd)
        opened_fds.append(leaf_fd)
        created_identity = os.fstat(leaf_fd)
        current_identity = os.stat(
            leaf,
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(current_identity.st_mode)
            or current_identity.st_dev != created_identity.st_dev
            or current_identity.st_ino != created_identity.st_ino
        ):
            raise OperationalTwinProtocolError(
                "candidate_attempt_claim_race: the candidate attempt leaf changed during claim."
            )
        checked = candidate_attempt_path(root, declared_output_root, require_absent=False)
        current_leaf = checked.lstat()
        if (current_leaf.st_dev, current_leaf.st_ino) != (created_identity.st_dev, created_identity.st_ino):
            raise OperationalTwinProtocolError("candidate_attempt_claim_race")
        return output_root
    except Exception:
        if leaf_created:
            _remove_created_candidate_leaf(
                parent_fd,
                declared_path.parts[-1],
                created_identity,
            )
        raise
    finally:
        for descriptor in reversed(opened_fds):
            os.close(descriptor)


def require_candidate_assignment_preflight(
    config: Mapping[str, object],
    root: Path,
    *,
    candidate_lineage_uid: str,
    protocol_uid: str,
    assignment_uid: str,
    attempt_uid: str,
    method_id: str,
    seed: int,
    output_root: str | Path,
    candidate_manifest_hash: str,
    assignment_manifest_hash: str,
    command_catalog_hash_value: str,
    preflight_uid: str,
    state_inputs: Mapping[str, object],
    require_recovery_predecessor: bool = True,
    resolved_context: ResolvedCandidateExecutionProtocol | None = None,
) -> dict[str, object]:
    protocol = _candidate_protocol(config, resolved_context)
    initialization = candidate_fresh_initialization_receipt(protocol, resolved_context)
    for field in CANDIDATE_FORBIDDEN_STATE_INPUT_FIELDS:
        if state_inputs.get(field) not in {None, False, ""}:
            raise OperationalTwinProtocolError(
                f"candidate_forbidden_state_input: {field} is forbidden for a fresh candidate lineage."
            )
    candidate_manifest, assignment_manifest, catalog, preflight = load_candidate_protocol_artifacts(config, root, resolved_context=resolved_context)
    blockers = validate_candidate_artifacts(
        config,
        root,
        candidate_manifest,
        assignment_manifest,
        catalog,
        preflight,
        require_attempt_roots_absent=False,
        resolved_context=resolved_context,
    )
    supplied = {
        "candidate_lineage_uid": candidate_lineage_uid,
        "protocol_uid": protocol_uid,
        "preflight_uid": preflight_uid,
        "candidate_manifest_hash": candidate_manifest_hash,
        "assignment_manifest_hash": assignment_manifest_hash,
        "command_catalog_hash": command_catalog_hash_value,
    }
    expected = {
        "candidate_lineage_uid": candidate_manifest.get("candidate_lineage_uid"),
        "protocol_uid": candidate_manifest.get("protocol_uid"),
        "preflight_uid": candidate_manifest.get("preflight_uid"),
        "candidate_manifest_hash": candidate_manifest.get("candidate_manifest_hash"),
        "assignment_manifest_hash": assignment_manifest.get("assignment_manifest_hash"),
        "command_catalog_hash": catalog.get("command_catalog_hash"),
    }
    for field, expected_value in expected.items():
        if supplied[field] != expected_value:
            blockers.append(_block("candidate_runtime_identity_mismatch", f"Runtime {field} drifted."))
    if candidate_manifest.get("initialization") != initialization:
        blockers.append(
            _block(
                "candidate_initialization_receipt_mismatch",
                "Candidate manifest fresh-initialization receipt drifted.",
            )
        )
    if assignment_manifest.get("initialization") != initialization:
        blockers.append(
            _block(
                "candidate_initialization_receipt_mismatch",
                "Assignment manifest fresh-initialization receipt drifted.",
            )
        )
    matches = [
        dict(row)
        for row in assignment_manifest.get("assignments", [])
        if dict(row).get("assignment_uid") == assignment_uid
    ]
    if len(matches) != 1:
        blockers.append(_block("candidate_assignment_uid_invalid", "Assignment UID is not present exactly once."))
        assignment: dict[str, object] = {}
        attempt: dict[str, object] = {}
    else:
        assignment = matches[0]
        if assignment.get("method_id") != method_id or int(assignment.get("seed", -1)) != int(seed):
            blockers.append(_block("candidate_assignment_identity_mismatch", "Method or seed differs from the frozen assignment."))
        attempts = [
            dict(row)
            for row in assignment.get("attempts", [])
            if dict(row).get("attempt_uid") == attempt_uid
        ]
        if len(attempts) != 1:
            blockers.append(_block("candidate_attempt_uid_invalid", "Attempt UID is not manifest-provided exactly once."))
            attempt = {}
        else:
            attempt = attempts[0]
            runtime_identity = candidate_identity_payload(assignment, attempt)
            expected_runtime_identity = {
                "candidate_lineage_uid": candidate_lineage_uid,
                "protocol_uid": protocol_uid,
                "assignment_uid": assignment_uid,
                "attempt_uid": attempt_uid,
            }
            if runtime_identity != expected_runtime_identity:
                blockers.append(
                    _block(
                        "candidate_runtime_identity_mismatch",
                        "Manifest assignment and attempt identities differ from runtime arguments.",
                    )
                )
            if (
                attempt.get("fresh_random_initialization") is not True
                or attempt.get("state_resume_allowed") is not False
            ):
                blockers.append(
                    _block(
                        "candidate_attempt_initialization_invalid",
                        "Attempt is not bound to fresh random initialization with state resume disabled.",
                    )
                )
            expected_output: Path | None = None
            supplied_output: Path | None = None
            try:
                expected_output = candidate_attempt_path(
                    root,
                    str(attempt.get("output_root", "")),
                    require_absent=True,
                )
            except OperationalTwinProtocolError as error:
                blockers.append(
                    _block("candidate_attempt_output_path_invalid", str(error))
                )
            try:
                supplied_output = candidate_attempt_path(
                    root,
                    output_root,
                    require_absent=False,
                )
            except OperationalTwinProtocolError as error:
                blockers.append(
                    _block("candidate_attempt_output_path_invalid", str(error))
                )
            if (
                expected_output is not None
                and supplied_output is not None
                and supplied_output != expected_output
            ):
                blockers.append(_block("candidate_attempt_output_root_mismatch", "Attempt output root differs from the manifest."))
            if bool(attempt.get("conditional")) and require_recovery_predecessor:
                predecessor_uid = str(attempt.get("predecessor_attempt_uid", ""))
                predecessors = [
                    dict(row)
                    for row in assignment.get("attempts", [])
                    if dict(row).get("attempt_uid") == predecessor_uid
                ]
                if len(predecessors) != 1:
                    blockers.append(_block("candidate_recovery_predecessor_missing", "Recovery predecessor is not frozen."))
                else:
                    predecessor = predecessors[0]
                    try:
                        predecessor_root = candidate_attempt_path(
                            root,
                            str(predecessor["output_root"]),
                            require_absent=False,
                        )
                    except OperationalTwinProtocolError as error:
                        blockers.append(
                            _block(
                                "candidate_recovery_predecessor_path_invalid",
                                str(error),
                            )
                        )
                        predecessor_root = None
                    status_path = (
                        predecessor_root / "terminal_status.json"
                        if predecessor_root is not None
                        else None
                    )
                    if status_path is None or not status_path.is_file():
                        blockers.append(_block("candidate_recovery_predecessor_status_missing", "Recovery requires a retained predecessor terminal status."))
                    else:
                        status = json.loads(status_path.read_text(encoding="utf-8"))
                        blockers.extend(
                            validate_candidate_recovery_predecessor(
                                protocol,
                                assignment,
                                predecessor,
                                status,
                                allowed_failure_class=attempt.get(
                                    "eligible_after_failure_class"
                                ),
                                root=root,
                                candidate_manifest=candidate_manifest,
                                assignment_manifest=assignment_manifest,
                                command_catalog=catalog,
                                preflight=preflight,
                                resolved_context=resolved_context,
                            )
                        )
    if not blockers and resolved_context is not None:
        from Libs.utils.evaluation.operational_twin_candidate_execution import require_candidate_runtime_ownership
        require_candidate_runtime_ownership(
            config, root, resolved_context, selected_attempt_uid=attempt_uid
        )
    if blockers:
        raise OperationalTwinProtocolError(
            "; ".join(f"{row['code']}: {row['message']}" for row in blockers)
        )
    return {
        "assignment": assignment,
        "attempt": attempt,
        "preflight": preflight,
        "initialization": initialization,
        "candidate_manifest": candidate_manifest,
        "assignment_manifest": assignment_manifest,
        "command_catalog": catalog,
        "resolved_protocol": resolved_context,
    }


def validate_assignment_runtime_evidence(
    assignment: Mapping[str, object],
    status: Mapping[str, object],
    provenance: Mapping[str, object],
    resolved_context: ResolvedCandidateExecutionProtocol | None = None,
) -> list[dict[str, str]]:
    """Validate terminal provenance against the exact frozen compute receipt."""
    blockers: list[dict[str, str]] = []
    assignment_uid = str(assignment.get("assignment_uid", ""))
    if status.get("status") != "COMPLETED":
        return blockers
    candidate_assignment = "candidate_lineage_uid" in assignment
    identity_fields = ["assignment_uid", "method_id", "seed"]
    if candidate_assignment:
        identity_fields = [
            "candidate_lineage_uid",
            "protocol_uid",
            "assignment_uid",
            "method_id",
            "seed",
        ]
    for field in identity_fields:
        if provenance.get(field) != assignment.get(field):
            blockers.append(
                _block(
                    "runtime_provenance_identity_mismatch",
                    f"{assignment_uid} runtime provenance does not match {field}.",
                )
            )
        if candidate_assignment and status.get(field) != provenance.get(field):
            blockers.append(
                _block(
                    "runtime_terminal_identity_mismatch",
                    f"{assignment_uid} terminal status does not match provenance {field}.",
                )
            )
    if candidate_assignment and status.get("attempt_uid") != provenance.get(
        "attempt_uid"
    ):
        blockers.append(
            _block(
                "runtime_terminal_identity_mismatch",
                f"{assignment_uid} terminal status does not match provenance attempt_uid.",
            )
        )
    if resolved_context is not None:
        for record in (status, provenance):
            for field in ("candidate_execution_authority", "candidate_resource_environment_receipt"):
                try:
                    require_successor_execution_binding(record.get(field, {}), resolved_context)
                except OperationalTwinProtocolError as error:
                    blockers.append(_block("runtime_successor_binding_invalid", str(error)))
    scheduler = provenance.get("scheduler_identity")
    if not isinstance(scheduler, Mapping) or not scheduler.get("slurm_job_id"):
        blockers.append(
            _block(
                "runtime_slurm_identity_missing",
                f"{assignment_uid} lacks a Slurm runtime identity.",
            )
        )
        scheduler = {}
    elif str(scheduler.get("hostname", "")).split(".", 1)[0] == "mgmt2":
        blockers.append(
            _block(
                "runtime_on_control_plane",
                f"{assignment_uid} reports execution on mgmt2.",
            )
        )
    device = provenance.get("device_provenance")
    if not isinstance(device, Mapping):
        blockers.append(
            _block(
                "runtime_device_provenance_missing",
                f"{assignment_uid} lacks device provenance.",
            )
        )
        return blockers
    resource_profile = str(assignment.get("resource_profile", ""))
    if device.get("resource_profile") != resource_profile:
        blockers.append(
            _block(
                "runtime_resource_profile_mismatch",
                f"{assignment_uid} did not bind the frozen resource profile.",
            )
        )
    requirements = assignment.get("resource_requirements")
    if candidate_assignment and requirements is not None:
        expected_requirements = CANDIDATE_RESOURCE_PROFILES.get(resource_profile)
        if requirements != expected_requirements:
            blockers.append(
                _block(
                    "runtime_resource_requirements_mismatch",
                    f"{assignment_uid} resource requirements drifted.",
                )
            )
            expected_requirements = dict(requirements)
        expected_cpus = int(expected_requirements.get("cpus_per_task", 0))
        try:
            observed_cpus = int(str(scheduler.get("slurm_cpus_per_task")))
        except ValueError:
            observed_cpus = -1
        observed_memory = _memory_mebibytes(scheduler.get("slurm_mem_per_node"))
        per_cpu_memory = _memory_mebibytes(scheduler.get("slurm_mem_per_cpu"))
        if observed_memory is None and per_cpu_memory is not None:
            observed_memory = per_cpu_memory * observed_cpus
        if (
            scheduler.get("slurm_job_partition")
            != expected_requirements.get("partition")
            or scheduler.get("slurm_job_nodelist")
            != expected_requirements.get("nodelist")
            or scheduler.get("hostname") != expected_requirements.get("nodelist")
            or device.get("slurm_partition")
            != expected_requirements.get("partition")
            or device.get("slurm_nodelist") != expected_requirements.get("nodelist")
            or device.get("hostname") != expected_requirements.get("nodelist")
            or observed_cpus != expected_cpus
            or observed_memory
            != _memory_mebibytes(expected_requirements.get("memory"))
        ):
            blockers.append(
                _block(
                    "runtime_exact_placement_mismatch",
                    f"{assignment_uid} did not use its exact frozen node and partition.",
                )
            )
        authority_binding = provenance.get("candidate_execution_authority")
        receipt_binding = provenance.get(
            "candidate_resource_environment_receipt"
        )
        if status.get("candidate_execution_authority") != authority_binding:
            blockers.append(
                _block(
                    "runtime_authority_binding_mismatch",
                    f"{assignment_uid} terminal authority binding differs from provenance.",
                )
            )
        if status.get("candidate_resource_environment_receipt") != receipt_binding:
            blockers.append(
                _block(
                    "runtime_resource_receipt_binding_mismatch",
                    f"{assignment_uid} terminal resource receipt differs from provenance.",
                )
            )
        if not isinstance(authority_binding, Mapping):
            blockers.append(
                _block(
                    "runtime_execution_authority_missing",
                    f"{assignment_uid} lacks execution-authority provenance.",
                )
            )
        elif set(authority_binding) != candidate_execution_binding_keys(CANDIDATE_EXECUTION_AUTHORITY_BINDING_KEYS, resolved_context):
            blockers.append(
                _block(
                    "runtime_execution_authority_binding_invalid",
                    f"{assignment_uid} execution-authority binding keyset drifted.",
                )
            )
        if not isinstance(receipt_binding, Mapping):
            blockers.append(
                _block(
                    "runtime_resource_receipt_missing",
                    f"{assignment_uid} lacks resource-receipt provenance.",
                )
            )
        else:
            if set(receipt_binding) != candidate_execution_binding_keys(CANDIDATE_RESOURCE_RECEIPT_BINDING_KEYS, resolved_context):
                blockers.append(
                    _block(
                        "runtime_resource_receipt_binding_invalid",
                        f"{assignment_uid} resource-receipt binding keyset drifted.",
                    )
                )
            if receipt_binding.get("resource_allocation") != expected_requirements:
                blockers.append(
                    _block(
                        "runtime_resource_receipt_allocation_mismatch",
                        f"{assignment_uid} resource allocation differs from the frozen profile.",
                    )
                )
            receipt_scheduler = receipt_binding.get("scheduler")
            receipt_environment = receipt_binding.get("environment")
            if not isinstance(receipt_scheduler, Mapping) or dict(
                receipt_scheduler
            ) != dict(scheduler):
                blockers.append(
                    _block(
                        "runtime_resource_receipt_scheduler_mismatch",
                        f"{assignment_uid} scheduler provenance differs from the bound receipt.",
                    )
                )
            if not isinstance(receipt_environment, Mapping):
                blockers.append(
                    _block(
                        "runtime_resource_receipt_environment_missing",
                        f"{assignment_uid} lacks the bound environment receipt.",
                    )
                )
            else:
                expected_environment = {
                    "hostname": device.get("hostname"),
                    "python_version": device.get("python_version"),
                    "torch_version": device.get("torch_version"),
                    "cuda_available": device.get("cuda_available"),
                    "cuda_device_count": device.get("cuda_device_count"),
                    "cuda_visible_devices": device.get("cuda_visible_devices"),
                }
                if dict(receipt_environment) != expected_environment:
                    blockers.append(
                        _block(
                            "runtime_resource_receipt_environment_mismatch",
                            f"{assignment_uid} device provenance differs from the bound receipt.",
                        )
                    )
    if resource_profile == "gpu_single":
        allocated = any(
            str(device.get(field, "")).strip()
            not in {"", "0", "(null)", "N/A", "None"}
            for field in (
                "slurm_job_gpus",
                "slurm_gpus_on_node",
                "slurm_step_gpus",
            )
        )
        if device.get("device_type") != "cuda" or device.get("cuda_available") is not True:
            blockers.append(
                _block(
                    "runtime_gpu_execution_missing",
                    f"{assignment_uid} did not execute with CUDA.",
                )
            )
        if candidate_assignment and (
            int(device.get("cuda_device_count") or 0) < 1
            or not str(device.get("cuda_visible_devices") or "").strip()
        ):
            blockers.append(
                _block(
                    "runtime_gpu_visibility_missing",
                    f"{assignment_uid} lacks an allocated visible CUDA device.",
                )
            )
        if candidate_assignment:
            job_gres = str(scheduler.get("slurm_job_gres") or "").strip()
            if job_gres and job_gres not in {"gpu:1", "gres/gpu:1"}:
                blockers.append(
                    _block(
                        "runtime_gpu_gres_mismatch",
                        f"{assignment_uid} Slurm GRES differs from gpu:1.",
                    )
                )
        if not allocated:
            blockers.append(
                _block(
                    "runtime_gpu_allocation_missing",
                    f"{assignment_uid} lacks explicit Slurm GPU allocation evidence.",
                )
            )
    elif resource_profile == "cpu_light" and candidate_assignment:
        allocated = any(
            str(device.get(field, "")).strip()
            not in {"", "0", "(null)", "N/A", "None"}
            for field in (
                "slurm_job_gpus",
                "slurm_gpus_on_node",
                "slurm_step_gpus",
            )
        )
        if (
            device.get("device_type") != "cpu"
            or device.get("cuda_available") is not False
            or int(device.get("cuda_device_count") or 0) != 0
            or str(device.get("cuda_visible_devices") or "").strip()
            or str(scheduler.get("slurm_job_gres") or "").strip()
            not in {"", "(null)", "N/A", "None"}
            or allocated
        ):
            blockers.append(
                _block(
                    "runtime_cpu_execution_mismatch",
                    f"{assignment_uid} is CPU-light but reports GPU allocation or CUDA execution.",
                )
            )
    elif resource_profile == "cpu_light" and device.get("device_type") != "cpu":
        blockers.append(
            _block(
                "runtime_cpu_execution_mismatch",
                f"{assignment_uid} is a CPU-light assignment but did not execute on CPU.",
            )
        )
    return blockers
